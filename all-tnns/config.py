from pydantic import BaseModel


class LayerInputParams:
    in_height: int
    in_width: int
    kernel_height: int
    kernel_width: int
    stride_h: int = 1
    stride_w: int = 1
    padding_h: int = 0
    padding_w: int = 0
    bias: bool = True


class ModelConfig(BaseModel):
    layer1: LayerInputParams = LayerInputParams(
        in_height=384, in_width=384, kernel_height=7, kernel_width=7
    )
    layer2: LayerInputParams = LayerInputParams(
        in_height=198, in_width=198, kernel_height=24, kernel_width=24
    )
    layer3: LayerInputParams = LayerInputParams(
        in_height=180, in_width=180, kernel_height=27, kernel_width=27
    )
    layer4: LayerInputParams = LayerInputParams(
        in_height=128, in_width=128, kernel_height=27, kernel_width=27
    )
    layer5: LayerInputParams = LayerInputParams(
        in_height=96, in_width=96, kernel_height=48, kernel_width=48
    )
    layer6: LayerInputParams = LayerInputParams(
        in_height=50, in_width=50, kernel_height=48, kernel_width=48
    )
    batch_size: int = 256
    fc1_hidden_dim: int = 565
    dropout_p: float = 0.2
    max_pool_kernel_dim: int = 2
    max_pool_stride: int = 2
