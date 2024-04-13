from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class LayerInputParams:
    kernel_dims: Tuple[int]
    in_dims: Tuple[int] = None
    num_kernels_out: Tuple[int] = None
    stride: Tuple[int] = (1,1)
    padding: Tuple[int] = (0,0)
    bias: bool = True
    in_channels: int = 1
    out_channels: int = 1


@dataclass
class ModelConfig:
    layers: List[LayerInputParams] = field(
        default_factory=lambda: [
        LayerInputParams(in_dims=(150,150), kernel_dims=(5,5), in_channels=3),
        LayerInputParams(kernel_dims=(7,7)),
        LayerInputParams(kernel_dims=(10,10)),
        LayerInputParams(kernel_dims=(13,13)),
        LayerInputParams(kernel_dims=(15,15)),
        LayerInputParams(kernel_dims=(1,1))
    ])
    batch_size: int = 256
    fc1_hidden_dim: int = 565
    dropout_p: float = 0.2
    max_pool_kernel_dim: int = 2
    max_pool_stride: int = 2

