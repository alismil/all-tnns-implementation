import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_similarity_loss_single_layer(
    weights: nn.ParameterList, layer_dims: tuple[int], alpha: float
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

    w = torch.stack([weight for weight in weights]).reshape(
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
    # num_kernels_h*root_channels, num_kernels_w*root_channels, in_channels*kernel_height*kernel_width

    horizontal_distances = cos(w[:, :-1, :], w[:, 1:, :])  # cos_dist(w_i,j, w_i,j+1)
    vertical_distances = cos(w[:-1, :, :], w[1:, :, :])  # cos_dist(w_i,j, w_i+1,j)

    n = len(weights) * out_channels  # number of activations in the 2D layer

    all_distances = torch.sum(horizontal_distances) + torch.sum(vertical_distances)
    return (alpha / (2 * n)) * all_distances


def spatial_similarity_loss(
    all_layer_weights: list[nn.ParameterList],
    all_layer_dims: list[tuple[int]],
    all_alpha: list[float],
):

    assert len(all_layer_weights) == len(all_layer_dims) == len(all_alpha)

    loss = []
    for i in range(len(all_layer_weights)):
        layer_loss = spatial_similarity_loss_single_layer(
            all_layer_weights[i], all_layer_dims[i], all_alpha[i]
        )
        loss.append(layer_loss)

    return sum(loss)


def cross_entropy_loss(output, target):
    loss = F.cross_entropy(output, target)
    return loss


def all_tnn_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    all_layer_weights: list[nn.ParameterList],
    all_layer_dims: list[tuple[int]],
    all_alpha: list[float],
):
    ce_loss = cross_entropy_loss(output, target)
    spatial_loss = spatial_similarity_loss(all_layer_weights, all_layer_dims, all_alpha)

    return ce_loss + spatial_loss
