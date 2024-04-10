import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from config import LayerInputParams, ModelConfig

"""
6 layers, of which layers 1, 3, and 5 are followed by 2 by 2 pooling layers. 
Each layer is followed by layer normalisation and a rectified linear unit. 
Weights are initialised with Xavier initialisation. 
A dropout of 0.2 is applied to all layers during training.
"""


class AllTnn(nn.Module):
    @staticmethod
    def calc_activation_shape(params: LayerInputParams):
        num_kernels_w = (
            math.floor(
                (params.in_width - params.kernel_width + 2 * params.padding_w)
                / params.stride_w
            )
            + 1
        )  # number of kernels that fit in the width of input
        num_kernels_h = (
            math.floor(
                (params.in_height - params.kernel_height + 2 * params.padding_h)
                / params.stride_h
            )
            + 1
        )  # number of kernels that fit in the height of input

        return num_kernels_w, num_kernels_h

    def __init__(self, config: ModelConfig):
        super().__init__()
        out_shape = AllTnn.calc_activation_shape(config.layer1)
        self.conv1 = LocallyConnected2dV1(
            params=config.layer1, num_kernels_w=out_shape[0], num_kernels_h=out_shape[1]
        )
        self.norm1 = torch.nn.LayerNorm([*out_shape])
        
        config.layer2.in_width = out_shape[0] // 2 # we will be applying 2x2 pooling
        config.layer2.in_height = out_shape[1] // 2
        out_shape = AllTnn.calc_activation_shape(config.layer2)
        self.conv2 = LocallyConnected2dV1(
            params=config.layer2, num_kernels_w=out_shape[0], num_kernels_h=out_shape[1]
        )
        self.norm2 = torch.nn.LayerNorm([*out_shape])

        config.layer3.in_width = out_shape[0]
        config.layer3.in_height = out_shape[1]
        out_shape = AllTnn.calc_activation_shape(config.layer3)
        self.conv3 = LocallyConnected2dV1(
            config.layer3, num_kernels_w=out_shape[0], num_kernels_h=out_shape[1]
        )
        self.norm3 = torch.nn.LayerNorm([*out_shape])

        config.layer4.in_width = out_shape[0] // 2
        config.layer4.in_height = out_shape[1] // 2
        out_shape = AllTnn.calc_activation_shape(config.layer4)
        self.conv4 = LocallyConnected2dV1(
            config.layer4, num_kernels_w=out_shape[0], num_kernels_h=out_shape[1]
        )
        self.norm4 = torch.nn.LayerNorm([*out_shape])

        config.layer5.in_width = out_shape[0]
        config.layer5.in_height = out_shape[1]
        out_shape = AllTnn.calc_activation_shape(config.layer5)
        self.conv5 = LocallyConnected2dV1(
            config.layer5, num_kernels_w=out_shape[0], num_kernels_h=out_shape[1]
        )
        self.norm5 = torch.nn.LayerNorm([*out_shape])

        config.layer6.in_width = out_shape[0] // 2
        config.layer6.in_height = out_shape[1] // 2
        out_shape = AllTnn.calc_activation_shape(config.layer6)
        self.conv6 = LocallyConnected2dV1(
            config.layer6, num_kernels_w=out_shape[0], num_kernels_h=out_shape[1]
        )
        self.norm6 = torch.nn.LayerNorm([*out_shape])

        self.pool = nn.MaxPool2d(config.max_pool_kernel_dim, config.max_pool_stride)
        self.dropout = nn.Dropout2d(p=config.dropout_p)
        self.fc1 = nn.Linear(out_shape[0] * out_shape[1], config.fc1_hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x))).unsqueeze(1)
        x = self.dropout(self.pool(x)).squeeze(1)
        x = self.dropout(self.norm2(F.relu(self.conv2(x))).unsqueeze(1)).squeeze(1)
        x = self.norm3(F.relu(self.conv3(x))).unsqueeze(1)
        x = self.dropout(self.pool(x)).squeeze(1)
        x = self.dropout(self.norm4(F.relu(self.conv4(x))).unsqueeze(1)).squeeze(1)
        x = self.norm5(F.relu(self.conv5(x))).unsqueeze(1)
        x = self.dropout(self.pool(x)).squeeze(1)
        x = self.dropout(self.norm6(F.relu(self.conv6(x))).unsqueeze(1)).squeeze(1)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return self.softmax(x)


class LocallyConnected2dV1(nn.Module):
    def __init__(self, params: LayerInputParams, num_kernels_w, num_kernels_h):
        super(LocallyConnected2dV1, self).__init__()
        assert params.kernel_height <= params.in_height + 2 * params.padding_h, f"kernel_height {params.kernel_height} is greater than max {params.in_height + 2 * params.padding_h}"
        assert params.kernel_width <= params.in_width + 2 * params.padding_w, f"kernel_width {params.kernel_width} is greater than max {params.in_width + 2 * params.padding_h}"

        self.in_height = params.in_height
        self.in_width = params.in_width
        self.stride_w = params.stride_w
        self.stride_h = params.stride_h
        self.padding_h = params.padding_h
        self.padding_w = params.padding_w
        self.kernel_width = params.kernel_width
        self.kernel_height = params.kernel_height
        self.num_kernels_w = num_kernels_w
        self.num_kernels_h = num_kernels_h
        num_kernels = self.num_kernels_w * self.num_kernels_h
        self.weights = [
            nn.Parameter(torch.Tensor(self.kernel_height, self.kernel_width))
            for _ in range(num_kernels)
        ]  # initialise the individual kernel weights

        if params.bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.num_kernels_h, self.num_kernels_w)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights: 
            nn.init.xavier_normal_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        # x of shape batch_size, in_height, in_width
        assert x.shape[1] == self.in_height, f"input height {x.shape[1]} does not equal expected height {self.in_height}"
        assert x.shape[2] == self.in_width, f"input width {x.shape[2]} does not equal expected width {self.in_width}"

        padding = (self.padding_w, self.padding_w, self.padding_h, self.padding_h)
        x = F.pad(x, padding, "constant", 0)

        out = torch.empty(x.shape[0], self.num_kernels_h, self.num_kernels_w)
        for i, weight in enumerate(self.weights):
            # find the top left corner index of the receptive field for the ith kernel
            start_i = (i // self.num_kernels_w) * self.stride_h
            start_j = (i % self.num_kernels_w) * self.stride_w
            in_patch = x[
                :,
                start_i : start_i + self.kernel_height,
                start_j : start_j + self.kernel_width,
            ]
            # convolve this cross-batch receptive field with the coresponding kernel and store the result
            conv = in_patch.mul(weight).sum((1, 2))
            out[:, i // self.num_kernels_w, i % self.num_kernels_w] = conv

        if self.bias is not None:
            out += self.bias

        return out


class LocallyConnected2dV2(nn.Module):
    def __init__(self, params: LayerInputParams, num_kernels_w, num_kernels_h):
        super(LocallyConnected2dV2, self).__init__()
        assert params.kernel_height <= params.in_height + 2 * params.padding_h
        assert params.kernel_width <= params.in_width + 2 * params.padding_w

        self.in_height = params.in_height
        self.in_width = params.in_width
        self.stride_w = params.stride_w
        self.stride_h = params.stride_h
        self.padding_h = params.padding_h
        self.padding_w = params.padding_w
        self.kernel_width = params.kernel_width
        self.kernel_height = params.kernel_height
        self.num_kernels_w = num_kernels_w
        self.num_kernels_h = num_kernels_h
        self.num_kernels = self.num_kernels_w * self.num_kernels_h
        self.kernel_weights = [
            nn.Parameter(torch.Tensor(self.kernel_height, self.kernel_width))
            for _ in range(self.num_kernels)
        ]  # initialise the individual kernel weights

        if params.bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.num_kernels_h, self.num_kernels_w)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.W = self._create_weight_matrix(self.kernel_weights)

    def reset_parameters(self):
        for weight in self.kernel_weights: 
            nn.init.xavier_normal_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _create_weight_matrix(self, weights):
        """
        Creates a sparse matrix which when muptiplied by the input
        gives the required untied weights for each receptive field.
        Number or rows corresponds to the number of input pixels and
        number of columns is the number total number of kernels that
        will fit in the padded input image when considering the stride.
        """

        init_w = torch.zeros(
            (self.in_height + 2 * self.padding_h)
            * (self.in_width + 2 * self.padding_w),
            self.num_kernels,
        )
        for i, weight in tqdm(enumerate(weights), total=len(weights)):
            # find the top left corner index of the ith kernel
            start_i = (i // self.num_kernels_w) * self.stride_h
            start_j = (i % self.num_kernels_w) * self.stride_w
            for k in range(self.kernel_height):
                # add each row of kernel weights to the sparse weight matrix
                begin = (
                    self.in_width + 2 * self.padding_w
                ) * start_i + start_j  # start index of flattened input for each kernel row
                init_w[begin : begin + self.kernel_width, i] = weight[k, :]
                start_i += 1

        return init_w

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.in_height
        assert x.shape[2] == self.in_width

        padding = (self.padding_w, self.padding_w, self.padding_h, self.padding_h)
        x = F.pad(x, padding, "constant", 0)

        x = x.view(
            x.shape[0],
            (self.in_height + 2 * self.padding_h)
            * (self.in_width + 2 * self.padding_w),
        )

        out = x @ self.W
        out = out.reshape(x.shape[0], self.num_kernels_h, self.num_kernels_w)

        if self.bias is not None:
            out += self.bias

        return out
