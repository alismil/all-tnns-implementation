import torch
import torch.nn as nn
import math

def spatial_similarity_loss_single_layer(weights: list[torch.Tensor], layer_dims: tuple[int], alpha: float):
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
        list_index = int((i // root_channels * layer_dims[1] / root_channels) + (j // root_channels))
        channel_index = int(((i % root_channels) * root_channels) + (j % root_channels))
        w = weights[list_index][channel_index, :, :, :]
        return torch.flatten(w, 1)

    loss = 0
    n = len(weights) * weights[0].shape[0]
    for i in range(layer_dims[0]-1):
        for j in range(layer_dims[1]-1):
            current_w = get_weight(i, j)
            right_w = get_weight(i, j + 1)
            bottom_w = get_weight(i + 1, j)
            print(f"i, j: {i}, {j}\n\n current_w: {current_w}\n\n right_w: {right_w}\n\n bottom_w: {bottom_w}")
            dist_current_right = cos(current_w, right_w)
            dist_current_bottom = cos(current_w, bottom_w)
            loss += dist_current_right + dist_current_bottom

    return (alpha/(2*n))*loss


num_kernels_h = 2
num_kernels_w = 2
out_channels = 4
root_channels = int(math.sqrt(out_channels))
in_channels = 1
kernel_height = 2
kernel_width = 2

weights = [torch.randint(3, (out_channels,in_channels,kernel_height,kernel_width), dtype=float) for _ in range(num_kernels_h * num_kernels_w)]
layer_dims = (num_kernels_h*root_channels,num_kernels_w*root_channels)
alpha = 0.1

print(weights)

print("loss: ", spatial_similarity_loss_single_layer(weights, layer_dims, alpha))
