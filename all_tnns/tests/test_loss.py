import torch

from all_tnns.loss_functions import spatial_similarity_loss_single_layer

weights = [
    torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [2.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [2.0, 2.0]]],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [[[1.0, 2.0], [1.0, 2.0]]],
            [[[1.0, 2.0], [0.0, 2.0]]],
            [[[2.0, 2.0], [2.0, 1.0]]],
            [[[0.0, 0.0], [2.0, 2.0]]],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 2.0]]],
            [[[1.0, 2.0], [1.0, 1.0]]],
            [[[0.0, 2.0], [1.0, 0.0]]],
            [[[1.0, 2.0], [1.0, 1.0]]],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [[[2.0, 2.0], [1.0, 2.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 2.0], [2.0, 0.0]]],
            [[[2.0, 1.0], [0.0, 2.0]]],
        ],
        dtype=torch.float64,
    ),
]

layer_dims = (4, 4)
alpha = 0.1


def test_spatial_similarity_layer_loss():
    loss = spatial_similarity_loss_single_layer(weights, layer_dims, alpha)
    assert round(loss.item(), 4) == 0.0551
