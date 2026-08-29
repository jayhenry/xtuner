import pytest
import torch

from xtuner.v1.ops.moe.cuda.group_gemm import triton_group_gemm
from xtuner.v1.ops.moe.cuda.route_weight import route_weight_rows_backward


@pytest.mark.parametrize("compile", [False, True])
def test_grouped_gemm_backward_returns_natural_bf16_weight_gradient(compile: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(17)
    counts = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    x = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = torch.cat(
        (
            x_ref[:2] @ weight_ref[0].T,
            x_ref[2:5] @ weight_ref[2].T,
            x_ref[5:] @ weight_ref[3].T,
        )
    )
    expected.backward(grad_output)

    grouped_gemm = triton_group_gemm
    if compile:
        grouped_gemm = torch.compile(grouped_gemm, fullgraph=True)
    actual = grouped_gemm(x, weight, counts)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(weight.grad[1], torch.zeros_like(weight.grad[1]), rtol=0, atol=0)


def test_fused_route_weight_backward_returns_bf16_rows_and_fp32_weights() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(23)
    grad_weighted = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    expert_output = torch.randn_like(grad_weighted)
    route_weights = torch.randn(7, device="cuda", dtype=torch.float32)

    grad_expert, grad_route = route_weight_rows_backward(
        grad_weighted,
        expert_output,
        route_weights,
    )
    # grouped-gemm's BF16 unpermute backward first rounds the FP32 router
    # weight to BF16, and its route-gradient dot rounds every product to BF16
    # before the FP32 reduction. MoonEP must preserve that public numerical
    # contract when it fuses the same operation into combine backward.
    expected_expert = grad_weighted * route_weights.bfloat16()[:, None]
    expected_route = (grad_weighted * expert_output).float().sum(dim=-1)

    assert grad_expert.dtype is torch.bfloat16
    assert grad_route.dtype is torch.float32
    torch.testing.assert_close(grad_expert, expected_expert, rtol=0, atol=0)
    torch.testing.assert_close(grad_route, expected_route, rtol=1e-5, atol=1e-4)
