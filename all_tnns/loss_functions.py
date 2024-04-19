import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_weight(i, j):
        list_index = int(
            (i // root_channels * layer_dims[1] / root_channels) + (j // root_channels)
        )
        channel_index = int(((i % root_channels) * root_channels) + (j % root_channels))
        w = weights[list_index][channel_index, :, :, :]
        return torch.flatten(w, 1)

    loss = []
    n = len(weights) * weights[0].shape[0]
    for i in range(layer_dims[0] - 1):
        for j in range(layer_dims[1] - 1):
            current_w = get_weight(i, j)
            right_w = get_weight(i, j + 1)
            bottom_w = get_weight(i + 1, j)
            dist_current_right = cos(current_w, right_w)
            dist_current_bottom = cos(current_w, bottom_w)
            loss.append(dist_current_right + dist_current_bottom)

    return (alpha / (2 * n)) * sum(loss)


def spatial_similarity_loss(
    all_layer_weights: list[list[torch.Tensor]],
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
    all_layer_weights: list[list[torch.Tensor]],
    all_layer_dims: list[tuple[int]],
    all_alpha: list[float],
):
    ce_loss = cross_entropy_loss(output, target)
    spatial_loss = spatial_similarity_loss(all_layer_weights, all_layer_dims, all_alpha)

    return ce_loss + spatial_loss
