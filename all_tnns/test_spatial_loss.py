import math
import time

import torch
import torch.nn as nn

# weights = [
#     torch.Tensor(
#         [
#             [[[1, 1], [1, 0]]],
#             [[[0, 1], [2, 1]]],
#             [[[0, 0], [0, 1]]],
#             [[[1, 1], [2, 2]]],
#         ],
#     ),
#     torch.Tensor(
#         [
#             [[[1, 2], [1, 2]]],
#             [[[1, 2], [0, 2]]],
#             [[[2, 2], [2, 1]]],
#             [[[0, 0], [2, 2]]],
#         ],
#     ),
#     torch.Tensor(
#         [
#             [[[0, 1], [2, 2]]],
#             [[[1, 2], [1, 1]]],
#             [[[0, 2], [1, 0]]],
#             [[[1, 2], [1, 1]]],
#         ],
#     ),
#     torch.Tensor(
#         [
#             [[[2, 2], [1, 2]]],
#             [[[1, 0], [0, 1]]],
#             [[[1, 2], [2, 0]]],
#             [[[2, 1], [0, 2]]],
#         ],
#     ),
# ]


def spatial_similarity_loss_single_layer(
    weights: list[torch.Tensor], layer_dims: tuple[int], alpha: float
):
    """
    Args:
        - weights: all the weights used in the given layer,
          each weight is out_channels, in_channels, kernel_height, kernel_width
          and there are num_kernels_h * num_kernels_w weights
        - layer_dims: (out_height, out_width)
        - alpha: constant for this layer multiplying the cosine distances
    """
    out_channels = weights[0].shape[0]
    root_channels = int(math.sqrt(out_channels))
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def get_weight(i, j):
        list_index = int(
            (i // root_channels * layer_dims[1] / root_channels) + (j // root_channels)
        )
        channel_index = int(((i % root_channels) * root_channels) + (j % root_channels))
        w = weights[list_index][channel_index, :, :, :]
        return w.reshape(1, -1)

    w_dim = weights[0].shape[1] * weights[0].shape[2] * weights[0].shape[3]
    w = torch.empty(layer_dims[0], layer_dims[1], w_dim)
    for i in range(layer_dims[0]):
        for j in range(layer_dims[1]):
            w[i, j, :] = get_weight(i, j)

    horizontal_distances = cos(w[:, :-1, :], w[:, 1:, :])  # cos_dist(w_i,j, w_i,j+1)
    vertical_distances = cos(w[:-1, :, :], w[1:, :, :])  # cos_dist(w_i,j, w_i+1,j)

    n = len(weights) * out_channels  # number of activations in the 2D layer

    all_distances = torch.sum(horizontal_distances) + torch.sum(vertical_distances)

    return (alpha / (2 * n)) * all_distances


def spatial_similarity_loss_single_layer_v2(
    weights: list[torch.Tensor], layer_dims: tuple[int], alpha: float
):
    """
    Args:
        - weights: all the weights used in the given layer,
          each weight is out_channels, in_channels, kernel_height, kernel_width
          and there are num_kernels_h * num_kernels_w weights
        - layer_dims: (out_height, out_width)
        - alpha: constant for this layer multiplying the cosine distances
    """
    out_channels = weights[0].shape[0]
    in_channels = weights[0].shape[1]
    kernel_height = weights[0].shape[2]
    kernel_width = weights[0].shape[3]

    root_channels = int(math.sqrt(out_channels))
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    w_dim = weights[0].shape[1] * weights[0].shape[2] * weights[0].shape[3]

    w = torch.stack(weights).reshape(
        layer_dims[0] // root_channels,
        layer_dims[1] // root_channels,
        out_channels,
        in_channels,
        kernel_height,
        kernel_width,
    )

    w = w.flatten(3, 5)
    # num_kernels_h, num_kernels_w, out_channels, in_channels*kernel_height*kernel_width

    w = w.permute(3, 0, 1, 2).reshape(
        w_dim,
        layer_dims[0] // root_channels,
        layer_dims[1] // root_channels,
        root_channels,
        root_channels,
    )
    w = w.permute(0, 1, 3, 2, 4).reshape(
        w_dim,
        layer_dims[0],
        layer_dims[1],
    )
    w = w.permute(1, 2, 0)

    horizontal_distances = cos(w[:, :-1, :], w[:, 1:, :])  # cos_dist(w_i,j, w_i,j+1)
    vertical_distances = cos(w[:-1, :, :], w[1:, :, :])  # cos_dist(w_i,j, w_i+1,j)

    n = len(weights) * out_channels  # number of activations in the 2D layer

    all_distances = torch.sum(horizontal_distances) + torch.sum(vertical_distances)

    return (alpha / (2 * n)) * all_distances


layer_dims = (384, 384)
alpha = 0.1

in_channels = 1
kernel_height, kernel_width = 7, 7
num_kernels_h, num_kernels_w = 48, 48
out_channels = 64

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

one = time.time()
l2 = spatial_similarity_loss_single_layer_v2(weights, layer_dims, alpha)
two = time.time()
l1 = spatial_similarity_loss_single_layer(weights, layer_dims, alpha)
three = time.time()

print(two - one, three - two)
print(l2, l1)
