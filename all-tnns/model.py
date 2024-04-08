import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_height, in_width, kernel_height, kernel_width, stride=1, bias=False):
        super(LocallyConnected2dV1, self).__init__()
        assert kernel_height <= in_height
        assert kernel_width <= in_width

        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.num_kernels_w = in_width - kernel_width + 1
        self.num_kernels_h = in_height - kernel_height + 1
        num_kernels = self.num_kernels_w * self.num_kernels_h
        self.weights = [nn.Parameter(torch.randn(kernel_height * kernel_width)) for _ in range(num_kernels)]
        
    def forward(self, x: torch.Tensor):

        out = []
        for i, weight in enumerate(self.weights):
            start_i, start_j = i // self.num_kernels_w, i % self.num_kernels_w
            in_patch = x[start_i:start_i + self.kernel_height, start_j:start_j + self.kernel_width]
            in_patch = in_patch.reshape(self.kernel_height * self.kernel_width)
            out.append(in_patch.dot(weight))

        return torch.tensor(out).reshape(self.num_kernels_h, self.num_kernels_w)
    

class LocallyConnected2dV2(nn.Module):
    def __init__(self, in_height, in_width, kernel_height, kernel_width, stride=1, bias=False):
        super(LocallyConnected2dV2, self).__init__()
        assert kernel_height <= in_height
        assert kernel_width <= in_width

        self.in_height = in_height
        self.in_width = in_width
        self.num_kernels_w = in_width - kernel_width + 1
        self.num_kernels_h = in_height - kernel_height + 1
        num_kernels = self.num_kernels_w * self.num_kernels_h

        weights = [nn.Parameter(torch.randn(kernel_height, kernel_width)) for _ in range(num_kernels)]
        self.W = torch.zeros(in_height*in_width, num_kernels)

        for i, weight in enumerate(weights):
            start_i, start_j = i // self.num_kernels_w, i % self.num_kernels_w
            for k in range(kernel_height):
                begin = in_width * start_i + start_j
                self.W[begin:begin+kernel_width, i] = weight[k,:]
                start_i += 1
        
    def forward(self, x: torch.Tensor):
        assert x.shape[0] == self.in_height
        assert x.shape[1] == self.in_width

        x = x.view(self.in_height*self.in_width)

        out = x @ self.W

        return out.reshape(self.num_kernels_h, self.num_kernels_w)
    
import time
x = torch.rand(3,4)

lc2d1 = LocallyConnected2dV1(in_height=3, in_width=4, kernel_height=2, kernel_width=2)
lc2d2 = LocallyConnected2dV2(in_height=3, in_width=4, kernel_height=2, kernel_width=2)

one = time.time()

lc2d1(x)

two = time.time()

lc2d2(x)

three = time.time()

print(two-one, three-two)
