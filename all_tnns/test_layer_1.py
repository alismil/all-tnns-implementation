import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_into_2D(x: torch.Tensor):
    batch_size, out_channels, out_height, out_width = x.shape
    root_channels = int(math.sqrt(out_channels))
    x = x.permute(0, 2, 3, 1).reshape(
        batch_size, out_height, out_width, root_channels, root_channels
    )
    x = x.permute(0, 1, 3, 2, 4).reshape(
        batch_size, root_channels * out_height, root_channels * out_width
    )
    return x.unsqueeze(1)


batch_size = 256
in_channels = 3
kernel_height, kernel_width = 7, 7
in_height, in_width = 150, 150
stride_h, stride_w = 3, 3
num_kernels_h, num_kernels_w = 48, 48
out_channels = 64
root_channels = int(math.sqrt(out_channels))

x = torch.rand(batch_size, in_channels, in_height, in_width)

weights = [
    nn.Parameter(
        torch.Tensor(
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        )
    )
    for _ in range(num_kernels_w * num_kernels_h)
]

start = time.time()

out = torch.empty(batch_size, out_channels, num_kernels_h, num_kernels_w)

for i, weight in enumerate(weights):
    start_i = (i // num_kernels_w) * stride_h
    start_j = (i % num_kernels_w) * stride_w
    in_patch = x[
        :,
        :,
        start_i : start_i + kernel_height,
        start_j : start_j + kernel_width,
    ]
    conv = F.conv2d(in_patch, weight, padding=0).squeeze()
    out[:, :, i // num_kernels_w, i % num_kernels_w] = conv

flatten_into_2D(out)

end = time.time()

print(end - start)
