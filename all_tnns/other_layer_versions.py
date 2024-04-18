import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from config import LayerInputParams
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    # TODO: add padding, Xavier init, dim checking, and update args to LayerInputParams
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def flatten_into_2D(self, x: torch.Tensor):
        # TODO add batch dim
        # batch x out_channels x out_height x out_width
        out_channels, out_height, out_width = x.shape
        root_channels = int(math.sqrt(out_channels))
        x = x.permute(1, 2, 0).reshape(out_height,out_width,root_channels,root_channels)
        x = x.permute(0, 2, 1, 3).reshape(root_channels*out_height, root_channels*out_width)
        return x
        
    def forward(self, x: torch.Tensor):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class LocallyConnected2dV2(nn.Module):
    def __init__(self, params: LayerInputParams):
        super(LocallyConnected2dV2, self).__init__()
        assert params.kernel_dims[0] <= params.in_dims[0] + 2 * params.padding[0]
        assert params.kernel_dims[1] <= params.in_dims[1] + 2 * params.padding[1]

        self.in_height, self.in_width = params.in_dims
        self.stride_h, self.stride_w = params.stride
        self.padding_h, self.padding_w = params.padding
        self.kernel_height, self.kernel_width = params.kernel_dims
        self.num_kernels_h, self.num_kernels_w = params.num_kernels_out
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
