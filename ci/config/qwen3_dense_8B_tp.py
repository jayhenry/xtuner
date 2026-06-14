import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.train import TrainerConfig


QWEN3_DENSE_PATH = os.environ["QWEN3_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]
TP_SIZE = int(os.environ.get("TP_SIZE", "2"))


dense_cfg = Qwen3Dense8BConfig(tp_size=8)
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=4098),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=4096)

loss_cfg = CELossConfig()


trainer = TrainerConfig(
    load_from=QWEN3_DENSE_PATH,
    model_cfg=dense_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_DENSE_PATH,
    global_batch_size=1,
    total_step=1000,
    work_dir="/tmp/qwen3_dense_8B_tp",
    seed=0,
)
