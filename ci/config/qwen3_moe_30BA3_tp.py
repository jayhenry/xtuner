"""Qwen3 MoE 30BA3 config with expert TP enabled.

Layout for 8 GPUs:
    fsdp_size = 1, ep_size = 4, expert_tp_size = 2

The GroupedLinear column-parallel weight (fused_w1w3) is built as a DTensor with
``(Shard, InterleavedShard)`` placement on the (ep, tp) sub-mesh; the post-FSDP layout
gets the standard FSDP-prepended ``_StridedShard`` on top.
"""

import os

from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig


QWEN3_MOE_PATH = "/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B"
ALPACA_PATH = "/mnt/shared-storage-user/llmrazor-share/data/alpaca"


moe_cfg = Qwen3MoE30BA3Config(
    ep_size=2,
    expert_tp_size=4,
    dispatcher="deepep",
)
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=True,
    cpu_offload=False,
    ep_size=moe_cfg.ep_size,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=8192),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=16384)

# loss_cfg = CELossConfig(mode="chunk")
loss_cfg = CELossConfig(mode="chunk")


trainer = TrainerConfig(
    load_from=QWEN3_MOE_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_MOE_PATH,
    global_batch_size=32,
    total_step=1000,
    work_dir="/mnt/shared-storage-user/llmrazor-share/yehaochen/tmp/",
    seed=0,
    # profile_step=5,
    # profile_memory=True,
    intra_layer_micro_batch=2,
)
