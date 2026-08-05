import os
import subprocess
import sys
import textwrap

import pytest
import torch


def test_many_document_packing_mask_compiles_without_expanding_each_document() -> None:
    script = textwrap.dedent(
        """
        import torch

        from xtuner.v1.ops.attn_imp import create_packing_block_causal_mask

        cu_seqlens = torch.arange(257, dtype=torch.int32) * 256

        def build_mask(cu_seqlens):
            return create_packing_block_causal_mask(cu_seqlens, sequence_length=65536).kv_num_blocks

        compiled = torch.compile(build_mask, backend="eager", fullgraph=True)
        assert compiled(cu_seqlens).numel() > 0

        mask = create_packing_block_causal_mask(cu_seqlens, sequence_length=65536)
        assert bool(mask.mask_mod(0, 0, 255, 0))
        assert not bool(mask.mask_mod(0, 0, 256, 255))
        assert not bool(mask.mask_mod(0, 0, 0, 1))
        """
    )

    # A subprocess gives graph capture a hard upper bound. The old Python loop
    # expanded 256 torch.full calls and spent minutes in Dynamo/SymPy here.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30, env=env)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
def test_flex_attention_runs_inside_a_compiled_caller() -> None:
    from xtuner.v1.ops.attn_imp import flex_attention

    # A single tiny kernel does not need Inductor's 32-process compile pool;
    # avoiding it also keeps the test's process lifecycle deterministic.
    torch._inductor.config.compile_threads = 1
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, 256], device="cuda", dtype=torch.int32)

    compiled = torch.compile(flex_attention)
    outputs = compiled(q, k, v, cu_seqlens)

    assert outputs["raw_output"].shape == (1, 256, 2, 64)
    assert torch.isfinite(outputs["raw_output"]).all()
