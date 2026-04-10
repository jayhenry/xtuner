"""Replay the generated k-path Triton kernel on saved probe tensors."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import triton


def _load_output_code(path: str):
    spec = importlib.util.spec_from_file_location("mha_kpath_output_code", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import output_code from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    d = (a.float() - b.float()).abs()
    return int((d > 0).sum().item()), float(d.max().item())


def _reference(
    buf11: torch.Tensor,
    mm1: torch.Tensor,
    norm_weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> torch.Tensor:
    x = mm1.float()
    dy = buf11.float()
    weight = norm_weight.float().view(1, 1, 1, 256) + 1.0
    rstd = rsqrt.float()
    dy_weight = dy * weight
    dot = (dy_weight * x).sum(dim=-1, keepdim=True)
    out = dy_weight * rstd + (-0.5 * dot * rstd * rstd * rstd / 256.0) * (2.0 * x)
    return out.to(torch.bfloat16)


def _set_triton_allocator(device: torch.device) -> None:
    if not hasattr(triton, "set_allocator"):
        return

    def alloc_fn(size: int, align: int, stream: int | None):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def _run_one(
    kernel,
    launcher,
    stream: int,
    mode: str,
    buf11: torch.Tensor,
    mm1: torch.Tensor,
    norm_weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> torch.Tensor:
    if mode == "inplace":
        out = mm1.clone()
        run_args = (out, buf11, norm_weight, rsqrt, 131072, 256)
    else:
        out = torch.empty_like(mm1)
        run_args = (buf11, norm_weight, mm1, rsqrt, out, 131072, 256)
    args_with_constexprs = kernel._get_args_with_constexprs(run_args, launcher)
    launcher(*args_with_constexprs, stream=stream)
    torch.cuda.synchronize()
    return out.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-code", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--mode", choices=("inplace", "outplace"), default="inplace")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Bypass autotune and run every precompiled Triton launcher once.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _set_triton_allocator(device)
    module = _load_output_code(args.output_code)
    kernel = module.triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5
    stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())

    base = Path(args.save_dir)
    prefix = f"rank{args.rank}_"
    buf11 = torch.load(base / f"{prefix}buf11_input.pt", map_location="cpu").to(args.device)
    mm1_path = base / f"{prefix}buf24_before.pt"
    if not mm1_path.exists():
        mm1_path = base / f"{prefix}mm1_input.pt"
    mm1 = torch.load(mm1_path, map_location="cpu").to(args.device)
    if mm1.ndim == 2 and mm1.shape[1] == 512:
        mm1 = mm1.view(1, mm1.shape[0], 2, 256)
    norm_weight = torch.load(base / f"{prefix}norm_weight.pt", map_location="cpu").to(args.device)
    rsqrt = torch.load(base / f"{prefix}rsqrt_1.pt", map_location="cpu").to(args.device)

    ref = _reference(buf11, mm1, norm_weight, rsqrt)
    ref_cpu = ref.cpu()
    if args.all_configs:
        kernel.precompile()
        outputs: list[torch.Tensor] = []
        for idx, launcher in enumerate(kernel.launchers):
            config = getattr(launcher, "config", None)
            out_cpu = _run_one(
                kernel,
                launcher,
                stream,
                args.mode,
                buf11,
                mm1,
                norm_weight,
                rsqrt,
            )
            outputs.append(out_cpu)
            n_ref, max_ref = _diff(out_cpu, ref_cpu)
            print(
                f"launcher {idx}: kwargs={getattr(config, 'kwargs', None)} "
                f"num_warps={getattr(config, 'num_warps', None)} "
                f"num_stages={getattr(config, 'num_stages', None)} "
                f"vs_ref n_differ={n_ref}/{out_cpu.numel()} "
                f"max_abs_diff={max_ref:.4e}"
            )
        first = outputs[0]
        for idx, out in enumerate(outputs[1:], start=1):
            n_diff, max_diff = _diff(out, first)
            print(
                f"launcher {idx}: vs_launcher0 n_differ={n_diff}/{out.numel()} "
                f"max_abs_diff={max_diff:.4e}"
            )
        return

    outputs: list[torch.Tensor] = []
    for idx in range(args.iters):
        if args.mode == "inplace":
            out = mm1.clone()
            kernel.run(out, buf11, norm_weight, rsqrt, 131072, 256, stream=stream)
        else:
            out = torch.empty_like(mm1)
            kernel.run(buf11, norm_weight, mm1, rsqrt, out, 131072, 256, stream=stream)
        torch.cuda.synchronize()
        out_cpu = out.cpu()
        outputs.append(out_cpu)
        n_ref, max_ref = _diff(out_cpu, ref_cpu)
        if idx == 0:
            launcher = kernel.launchers[0] if getattr(kernel, "launchers", None) else None
            config = getattr(launcher, "config", None)
            print(
                "target_config "
                f"kwargs={getattr(config, 'kwargs', None)} "
                f"num_warps={getattr(config, 'num_warps', None)} "
                f"num_stages={getattr(config, 'num_stages', None)}"
            )
        print(
            f"iter {idx}: vs_ref n_differ={n_ref}/{out_cpu.numel()} "
            f"max_abs_diff={max_ref:.4e}"
        )

    first = outputs[0]
    for idx, out in enumerate(outputs[1:], start=1):
        n_diff, max_diff = _diff(out, first)
        print(
            f"iter {idx}: vs_iter0 n_differ={n_diff}/{out.numel()} "
            f"max_abs_diff={max_diff:.4e}"
        )


if __name__ == "__main__":
    main()
