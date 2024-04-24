import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LayerInputParams, ModelConfig

"""
6 layers, of which layers 1, 3, and 5 are followed by 2 by 2 pooling layers. 
Each layer is followed by layer normalisation and a rectified linear unit. 
Weights are initialised with Xavier initialisation. 
A dropout of 0.2 is applied to all layers during training.
"""


class LocallyConnected2dV1(nn.Module):
    def __init__(self, params: LayerInputParams):
        super(LocallyConnected2dV1, self).__init__()
        assert (
            params.kernel_dims[0] <= params.in_dims[0] + 2 * params.padding[0]
        ), f"kernel_height {params.kernel_dims[0]} is greater than max {params.in_dims[0] + 2 * params.padding[0]}"
        assert (
            params.kernel_dims[1] <= params.in_dims[1] + 2 * params.padding[1]
        ), f"kernel_width {params.kernel_dims[1]} is greater than max {params.in_dims[1] + 2 * params.padding[1]}"
        assert (
            math.sqrt(params.out_channels) % 1 == 0.0
        ), f"out_channels is set to {params.out_channels}, which is not a perfect square"

        self.in_height, self.in_width = params.in_dims
        self.stride_h, self.stride_w = params.stride
        self.padding_h, self.padding_w = params.padding
        self.kernel_height, self.kernel_width = params.kernel_dims
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.num_kernels_h, self.num_kernels_w = params.num_kernels_out
        num_kernels = self.num_kernels_w * self.num_kernels_h

        self.weights = [
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_height,
                    self.kernel_width,
                )
            )
            for _ in range(num_kernels)
        ]  # initialise the individual kernel weights

        if params.bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels, self.num_kernels_h, self.num_kernels_w)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_normal_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def flatten_into_2D(self, x: torch.Tensor):
        """
        Takes a 3D layer of size channels x height x width per batch
        and 'unrolls' it into a 2D layer of size
        sqrt(channels)*height x sqrt(channels)*width per batch
        where e.g. a vector across the channel dimension for a
        particular height and width, say [1,2,3,4], gets represented as
        [[1,2,...],[3,4,...],...] in the 2D layer.
        """
        batch_size, out_channels, out_height, out_width = x.shape
        root_channels = int(math.sqrt(out_channels))
        x = x.permute(0, 2, 3, 1).reshape(
            batch_size, out_height, out_width, root_channels, root_channels
        )
        x = x.permute(0, 1, 3, 2, 4).reshape(
            batch_size, root_channels * out_height, root_channels * out_width
        )
        return x.unsqueeze(1)

    def forward(self, x: torch.Tensor):
        # x of shape batch_size, in_channels, in_height, in_width
        assert (
            x.shape[1] == self.in_channels
        ), f"number of input channels {x.shape[1]} does not equal expected {self.in_channels}"
        assert (
            x.shape[2] == self.in_height
        ), f"input height {x.shape[2]} does not equal expected height {self.in_height}"
        assert (
            x.shape[3] == self.in_width
        ), f"input width {x.shape[3]} does not equal expected width {self.in_width}"

        padding = (self.padding_w, self.padding_w, self.padding_h, self.padding_h)
        x = F.pad(x, padding, "constant", 0)

        out = torch.empty(
            x.shape[0], self.out_channels, self.num_kernels_h, self.num_kernels_w
        )
        for i, weight in enumerate(self.weights):
            # find the top left corner index of the receptive field for the ith kernel
            start_i = (i // self.num_kernels_w) * self.stride_h
            start_j = (i % self.num_kernels_w) * self.stride_w
            in_patch = x[
                :,
                :,
                start_i : start_i + self.kernel_height,
                start_j : start_j + self.kernel_width,
            ]
            # convolve this cross-batch receptive field with the coresponding kernel and store the result
            # in_patch: batch_size x in_channels x kernel_height x kernel_width
            # weight: out_channels x in_channels x kernel_height x kernel_width
            conv = F.conv2d(in_patch, weight, padding=0).squeeze()
            out[:, :, i // self.num_kernels_w, i % self.num_kernels_w] = conv

        if self.bias is not None:
            out += self.bias

        return self.flatten_into_2D(out), self.weights


class AllTnn(nn.Module):
    @staticmethod
    def calc_activation_shape(params: LayerInputParams):
        num_kernels_w = (
            math.floor(
                (params.in_dims[1] - params.kernel_dims[1] + 2 * params.padding[1])
                / params.stride[1]
            )
            + 1
        )  # number of kernels that fit in the width of input
        num_kernels_h = (
            math.floor(
                (params.in_dims[0] - params.kernel_dims[0] + 2 * params.padding[0])
                / params.stride[0]
            )
            + 1
        )  # number of kernels that fit in the height of input

        return int(math.sqrt(params.out_channels) * num_kernels_h), int(
            math.sqrt(params.out_channels) * num_kernels_w
        )

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.convs = []
        self.norms = []
        self.num_layers = len(config.layers)

        for i in range(self.num_layers):
            out_shape = AllTnn.calc_activation_shape(config.layers[i])
            config.layers[i].num_kernels_out = tuple(
                int(x // math.sqrt(config.layers[i].out_channels)) for x in out_shape
            )
            self.convs.append(LocallyConnected2dV1(config.layers[i]))
            self.norms.append(torch.nn.LayerNorm([*out_shape]))
            # we will be applying 2x2 pooling to every other layer
            div = 2 if i % 2 == 0 else 1
            if i < self.num_layers - 1:
                config.layers[i + 1].in_dims = tuple(x // div for x in out_shape)

        self.pool = nn.MaxPool2d(config.max_pool_kernel_dim, config.max_pool_stride)
        self.dropout = nn.Dropout2d(p=config.dropout_p)
        self.fc1 = nn.Linear(out_shape[0] * out_shape[1], config.fc1_hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):

        # keep track of all layer weights and dimensions for the spatial similarity loss
        all_weights = []
        all_layer_dims = []

        for i in range(self.num_layers):
            x, weights = self.convs[i](x)
            all_weights.append(weights)
            all_layer_dims.append((x.shape[2], x.shape[3]))
            x = self.norms[i](F.relu(x))
            if i % 2 == 0:
                x = self.pool(x)
            x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return self.softmax(x), all_weights, all_layer_dims
