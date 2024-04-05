import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, output_size, kernel_size, stride=1, out_channels=1, bias=False):
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
        
    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

"""
6 layers, of which layers 1, 3, and 5 are followed by 2 by 2 pooling layers. 
Each layer is followed by layer normalisation and a rectified linear unit. 
Weights are initialised with Xavier initialisation. 
A dropout of 0.2 is applied to all layers during training.
"""

class AllTnn(nn.Module):
    @staticmethod
    def calc_activation_shape(
        dim, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0)
    ):
        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            return (odim_i / stride[i]) + 1

        return shape_each_dim(0), shape_each_dim(1)
    
    def __init__(self, idim):
        super().__init__()
        self.conv1 = LocallyConnected2d(3, 384, 7)
        ln_shape = AllTnn.calc_activation_shape(idim, 7)
        self.norm1 = torch.nn.LayerNorm([1, *ln_shape])

        self.conv2 = LocallyConnected2d(1, 198, 24)
        ln_shape = AllTnn.calc_activation_shape(ln_shape, 24)
        self.norm2 = torch.nn.LayerNorm([1, *ln_shape])

        self.conv3 = LocallyConnected2d(1, 180, 27)
        ln_shape = AllTnn.calc_activation_shape(ln_shape, 27)
        self.norm3 = torch.nn.LayerNorm([1, *ln_shape])

        self.conv4 = LocallyConnected2d(1, 128, 27)
        ln_shape = AllTnn.calc_activation_shape(ln_shape, 27)
        self.norm4 = torch.nn.LayerNorm([1, *ln_shape])

        self.conv5 = LocallyConnected2d(1, 96, 48)
        ln_shape = AllTnn.calc_activation_shape(ln_shape, 48)
        self.norm5 = torch.nn.LayerNorm([1, *ln_shape])

        self.conv6 = LocallyConnected2d(1, 50, 48)
        ln_shape = AllTnn.calc_activation_shape(ln_shape, 48)
        self.norm6 = torch.nn.LayerNorm([1, *ln_shape])

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(48**2*(50-47)**2, 565)

    def forward(self, x):
        x = self.dropout(self.pool(self.norm1(F.relu(self.conv1(x)))))
        x = self.dropout(self.norm2(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(self.norm3(F.relu(self.conv3(x)))))
        x = self.dropout(self.norm4(F.relu(self.conv4(x))))
        x = self.dropout(self.pool(self.norm5(F.relu(self.conv5(x)))))
        x = self.dropout(self.norm6(F.relu(self.conv6(x))))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return nn.Softmax(x)
