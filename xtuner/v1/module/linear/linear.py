import os

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.float8.config import ScalingGranularity
from xtuner.v1.float8.float8_linear_tensor_wise import TensorWiseFloat8Linear
from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear


class _Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer."""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        return F.linear(input, w, b)


def build_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    device=None,
    dtype=None,
    float8_cfg=None,
) -> nn.Module:
    """Build a linear layer with optional float8 support."""
    if os.environ.get("XTUNER_USE_HIF8_CUDA", "0") == "1":
        from quant_cy import QType
        from quant_cy.layers.QLinear import QLinear

        # xTODO: TensorwiseHif8QLinear.pad_for_fsdp() 不需要做 pad，因为 scale 是 per-tensor 的
        # xTODO: TensorwiseHif8QLinear support module._precomputed_scale, 不做 pre compute,而是直接在 forward/backward 中做 scale 和 descale
        # xTODO: TensorwiseHif8QLinear support forward with scale and descale
        new_mod = QLinear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        qtype_str = "hif8"
        quant_type = QType(qtype_str)  # .dim(0)
        new_mod.assign_qparams(quant_type)
        quant_grad: bool = True
        new_mod.set_quant_grad(quant_grad)
        if os.environ.get("XTUNER_USE_HIF8_TENSORWISE_SCALE", "0") == "1":
            new_mod.tensorwise_scale = True
        print(
            f"Use HiF8 CUDA QLinear with tensorwise_scale: {new_mod.tensorwise_scale}, quant_grad: {quant_grad}, qtype: {quant_type}"
        )
        return new_mod

    if float8_cfg is None:
        return _Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif float8_cfg.scaling_granularity_gemm is ScalingGranularity.TILEWISE:
        return TileWiseFloat8Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif float8_cfg.scaling_granularity_gemm is ScalingGranularity.TENSORWISE:
        return TensorWiseFloat8Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Unsupported float8 scaling granularity: {float8_cfg.scaling_granularity_gemm}")
