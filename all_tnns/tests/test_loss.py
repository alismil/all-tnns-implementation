import torch
from all_tnns.loss_functions import spatial_similarity_loss_single_layer

weights = [
    torch.tensor([[[[1., 1.],
          [1., 0.]]],

        [[[0., 1.],
          [2., 1.]]],

        [[[0., 0.],
          [0., 1.]]],

        [[[1., 1.],
          [2., 2.]]]], dtype=torch.float64), 
          
    torch.tensor([[[[1., 2.],
          [1., 2.]]],

        [[[1., 2.],
          [0., 2.]]],

        [[[2., 2.],
          [2., 1.]]],

        [[[0., 0.],
          [2., 2.]]]], dtype=torch.float64),
          
    torch.tensor([[[[0., 1.],
          [2., 2.]]],

        [[[1., 2.],
          [1., 1.]]],

        [[[0., 2.],
          [1., 0.]]],

        [[[1., 2.],
          [1., 1.]]]], dtype=torch.float64), 
          
    torch.tensor([[[[2., 2.],
          [1., 2.]]],

        [[[1., 0.],
          [0., 1.]]],

        [[[1., 2.],
          [2., 0.]]],

        [[[2., 1.],
          [0., 2.]]]], dtype=torch.float64)
    ]

layer_dims = (4, 4)
alpha = 0.1

def test_spatial_similarity_layer_loss():
    loss = spatial_similarity_loss_single_layer(weights, layer_dims, alpha)
    assert round(loss.item, 4) == 0.0424
