from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import torch


@dataclass
class LayerInputParams:
    kernel_dims: Tuple[int]
    in_dims: Tuple[int] = None
    num_kernels_out: Tuple[int] = None
    stride: Tuple[int] = (1, 1)
    padding: Tuple[int] = (0, 0)
    bias: bool = True
    in_channels: int = 1
    out_channels: int = 1
    alpha: float = (
        1  # multiplicative parameter in the spatial similarity loss for the layer
    )


@dataclass
class ModelConfig:
    layers: List[LayerInputParams] = field(
        default_factory=lambda: [
            LayerInputParams(
                in_dims=(128, 128),
                kernel_dims=(7, 7),
                stride=(4, 4),
                padding=(2, 2),
                in_channels=3,
                out_channels=144,
            ),
            LayerInputParams(kernel_dims=(24, 24), stride=(8, 8), out_channels=81),
            LayerInputParams(kernel_dims=(27, 27), stride=(9, 9), out_channels=81),
            LayerInputParams(kernel_dims=(27, 27), stride=(4, 4), out_channels=64),
            LayerInputParams(kernel_dims=(48, 48), stride=(7, 7), out_channels=64),
            LayerInputParams(
                kernel_dims=(48, 48), stride=(1, 1), out_channels=2500, alpha=10
            ),
        ]
    )
    fc1_hidden_dim: int = 100
    dropout_p: float = 0.2
    max_pool_kernel_dim: int = 2
    max_pool_stride: int = 2


@dataclass
class TrainConfig:
    num_epochs: int = 10
    eval_interval: int = 50  # number of mini batches between evals
    log_interval: int = 10  # number of mini batches between logs
    eval_iters: int = 50  # number of mini batches to use for eval

    train_batch_size: int = 256
    val_batch_size: int = 100

    out_dir: str = "checkpoints"

    # data loading
    dataset_path: str = "clane9/imagenet-100"
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    input_size: int = 128
    min_scale: float = 0.4
    hflip: float = 0.5
    color_jitter: Optional[float] = 0.4
    interpolation: str = "bicubic"
    keep_in_memory: bool = False

    # optimizer
    lr: float = 0.001
    eps: float = 0.1
    weight_decay: float = 1e-6

    # wandb logging
    wandb_log = True
    wandb_project = "All-TNNs"
    wandb_run_name = f"{str(datetime.now()).replace(" ", "_")}_imagenet_100"

    # hardware
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
