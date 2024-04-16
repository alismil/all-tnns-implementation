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
        LayerInputParams(in_dims=(150,150), kernel_dims=(7,7), stride=(3,3), in_channels=3, out_channels=64),
        LayerInputParams(kernel_dims=(24,24), stride=(8,8), out_channels=81),
        LayerInputParams(kernel_dims=(27,27), stride=(9,9), out_channels=81),
        LayerInputParams(kernel_dims=(27,27), stride=(4,4), out_channels=64),
        LayerInputParams(kernel_dims=(48,48), stride=(7,7), out_channels=64),
        LayerInputParams(kernel_dims=(48,48), stride=(1,1), out_channels=2500)
    ])
    batch_size: int = 256
    fc1_hidden_dim: int = 565
    dropout_p: float = 0.2
    max_pool_kernel_dim: int = 2
    max_pool_stride: int = 2

