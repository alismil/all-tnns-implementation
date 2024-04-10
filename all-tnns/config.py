from dataclasses import dataclass


@dataclass
class LayerInputParams:
    kernel_height: int
    kernel_width: int
    in_height: int = None
    in_width: int = None
    stride_h: int = 1
    stride_w: int = 1
    padding_h: int = 0
    padding_w: int = 0
    bias: bool = True


@dataclass
class ModelConfig:
    layer1: LayerInputParams = LayerInputParams(
        in_height=150, in_width=150, kernel_height=5, kernel_width=5
    )
    layer2: LayerInputParams = LayerInputParams(
        kernel_height=7, kernel_width=7
    )
    layer3: LayerInputParams = LayerInputParams(
        kernel_height=10, kernel_width=10
    )
    layer4: LayerInputParams = LayerInputParams(
        kernel_height=13, kernel_width=13
    )
    layer5: LayerInputParams = LayerInputParams(
       kernel_height=15, kernel_width=15
    )
    layer6: LayerInputParams = LayerInputParams(
       kernel_height=1, kernel_width=1
    )
    batch_size: int = 256
    fc1_hidden_dim: int = 565
    dropout_p: float = 0.2
    max_pool_kernel_dim: int = 2
    max_pool_stride: int = 2
