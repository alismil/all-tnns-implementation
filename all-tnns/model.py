import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
6 layers, of which layers 1, 3, and 5 are followed by 2 by 2 pooling layers. 
Each layer is followed by layer normalisation and a rectified linear unit. 
Weights are initialised with Xavier initialisation. 
A dropout of 0.2 is applied to all layers during training.
"""

# class AllTnn(nn.Module):
#     @staticmethod
#     def calc_activation_shape(
#         dim, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0)
#     ):
#         def shape_each_dim(i):
#             odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
#             return (odim_i / stride[i]) + 1

#         return shape_each_dim(0), shape_each_dim(1)

#     def __init__(self, idim):
#         super().__init__()
#         self.conv1 = LocallyConnected2d(3, 384, 7)
#         ln_shape = AllTnn.calc_activation_shape(idim, 7)
#         self.norm1 = torch.nn.LayerNorm([1, *ln_shape])

#         self.conv2 = LocallyConnected2d(1, 198, 24)
#         ln_shape = AllTnn.calc_activation_shape(ln_shape, 24)
#         self.norm2 = torch.nn.LayerNorm([1, *ln_shape])

#         self.conv3 = LocallyConnected2d(1, 180, 27)
#         ln_shape = AllTnn.calc_activation_shape(ln_shape, 27)
#         self.norm3 = torch.nn.LayerNorm([1, *ln_shape])

#         self.conv4 = LocallyConnected2d(1, 128, 27)
#         ln_shape = AllTnn.calc_activation_shape(ln_shape, 27)
#         self.norm4 = torch.nn.LayerNorm([1, *ln_shape])

#         self.conv5 = LocallyConnected2d(1, 96, 48)
#         ln_shape = AllTnn.calc_activation_shape(ln_shape, 48)
#         self.norm5 = torch.nn.LayerNorm([1, *ln_shape])

#         self.conv6 = LocallyConnected2d(1, 50, 48)
#         ln_shape = AllTnn.calc_activation_shape(ln_shape, 48)
#         self.norm6 = torch.nn.LayerNorm([1, *ln_shape])

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d(p=0.2)
#         self.fc1 = nn.Linear(48**2*(50-47)**2, 565)

#     def forward(self, x):
#         x = self.dropout(self.pool(self.norm1(F.relu(self.conv1(x)))))
#         x = self.dropout(self.norm2(F.relu(self.conv2(x))))
#         x = self.dropout(self.pool(self.norm3(F.relu(self.conv3(x)))))
#         x = self.dropout(self.norm4(F.relu(self.conv4(x))))
#         x = self.dropout(self.pool(self.norm5(F.relu(self.conv5(x)))))
#         x = self.dropout(self.norm6(F.relu(self.conv6(x))))

#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = self.fc1(x)
#         return nn.Softmax(x)


class LocallyConnected2dV1(nn.Module):
    def __init__(
        self, in_height, in_width, kernel_height, kernel_width, stride=1, bias=False
    ):
        super(LocallyConnected2dV1, self).__init__()
        assert kernel_height <= in_height
        assert kernel_width <= in_width

        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.num_kernels_w = math.floor((in_width - kernel_width) / stride) + 1
        self.num_kernels_h = math.floor((in_height - kernel_height) / stride) + 1
        num_kernels = self.num_kernels_w * self.num_kernels_h
        self.weights = [
            nn.Parameter(torch.randn(kernel_height * kernel_width))
            for _ in range(num_kernels)
        ]

    def forward(self, x: torch.Tensor):

        out = []
        for i, weight in enumerate(self.weights):
            start_i, start_j = i // self.num_kernels_w, i % self.num_kernels_w
            in_patch = x[
                start_i : start_i + self.kernel_height,
                start_j : start_j + self.kernel_width,
            ]
            in_patch = in_patch.reshape(self.kernel_height * self.kernel_width)
            out.append(in_patch.dot(weight))

        return torch.tensor(out).reshape(self.num_kernels_h, self.num_kernels_w)


class LocallyConnected2dV2(nn.Module):
    def __init__(
        self,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride_h=1,
        stride_w=1,
        padding_h=0,
        padding_w=0,
        bias=False,
    ):
        super(LocallyConnected2dV2, self).__init__()
        assert kernel_height <= in_height + 2 * padding_h
        assert kernel_width <= in_width + 2 * padding_w

        self.in_height = in_height
        self.in_width = in_width
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.in_height = in_height
        self.in_width = in_width
        self.num_kernels_w = (
            math.floor((in_width - kernel_width + 2 * self.padding_w) / stride_w) + 1
        )  # number of kernels that fit in the width of input
        self.num_kernels_h = (
            math.floor((in_height - kernel_height + 2 * self.padding_h) / stride_h) + 1
        )  # number of kernels that fit in the height of input
        print("*******", self.num_kernels_h, self.num_kernels_w)
        self.num_kernels = self.num_kernels_w * self.num_kernels_h

        kernel_weights = [
            nn.Parameter(torch.randn(kernel_height, kernel_width))
            for _ in range(self.num_kernels)
        ]  # initialise the individual kernel weights

        self.W = self._create_weight_matrix(kernel_weights)

        if bias:
            self.bias = nn.Parameter(
                torch.randn(self.num_kernels_h, self.num_kernels_w)
            )
        else:
            self.register_parameter("bias", None)

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
        for i, weight in enumerate(weights):
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
        assert x.shape[0] == self.in_height
        assert x.shape[1] == self.in_width

        padding = (self.padding_w, self.padding_w, self.padding_h, self.padding_h)
        x = F.pad(x, padding, "constant", 0)

        x = x.view(
            (self.in_height + 2 * self.padding_h) * (self.in_width + 2 * self.padding_w)
        )

        out = x @ self.W

        out = out.reshape(self.num_kernels_h, self.num_kernels_w)

        if self.bias is not None:
            out += self.bias

        return out


# import time
# x = torch.rand(3,4)

# lc2d1 = LocallyConnected2dV1(in_height=3, in_width=4, kernel_height=2, kernel_width=2)
# lc2d2 = LocallyConnected2dV2(in_height=3, in_width=4, kernel_height=2, kernel_width=2)

# one = time.time()

# lc2d1(x)

# two = time.time()

# lc2d2(x)

# three = time.time()

# print(two-one, three-two)


x = torch.rand(450, 450)

lc2d2 = LocallyConnected2dV2(
    in_height=450,
    in_width=450,
    kernel_height=7,
    kernel_width=7,
    stride_h=1,
    stride_w=1,
    padding_h=0,
    padding_w=0,
)

lc2d2(x)
