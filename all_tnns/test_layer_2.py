import math
import time

import torch
import torch.nn as nn

batch_size = 256
in_channels = 3
kernel_height, kernel_width = 7, 7
in_height, in_width = 150, 150
stride_h, stride_w = 3, 3
num_kernels_h, num_kernels_w = 48, 48
out_channels = 64
root_channels = int(math.sqrt(out_channels))

x = torch.rand(batch_size, in_channels, in_height, in_width)

weights_proposed = nn.Parameter(
    torch.Tensor(
        root_channels * num_kernels_h,
        root_channels * num_kernels_w,
        kernel_height,
        kernel_width,
        in_channels,
    )
)

start = time.time()

x = x.unfold(2, kernel_height, stride_h).unfold(3, kernel_width, stride_w)
x = x.permute(
    0, 2, 3, 4, 5, 1
)  # batch_size, num_kernels_h, num_kernels_w, kernel_height, kernel_width, in_channels
x = x.repeat_interleave(repeats=root_channels, dim=1).repeat_interleave(
    repeats=root_channels, dim=2
)
# batch_size, num_kernels_h*root_channels, num_kernels_w*root_channels, kernel_height, kernel_width, in_channels
x = weights_proposed.mul(x).sum((3, 4, 5))

end = time.time()
print(end - start)
