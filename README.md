# all-tnns-implementation

A PyTorch implementation of the All-TNNs model as proposed in:

Lu, Zejin, et al. ["End-to-end topographic networks as models of cortical map formation and human visual behaviour: moving beyond convolutions."](https://arxiv.org/pdf/2308.09431.pdf) *arXiv preprint arXiv:2308.09431* (2023).

# Installation

This project uses Poetry for dependency management. To install Poetry run

```
curl -sSL https://install.python-poetry.org | python3 -
```

Then from the project root directory run

```
poetry shell
poetry install
```

All the model and train configurations should be set in `all_tnns/config.py`. 

To train the model run the following from the project root directory:

```
python3 all_tnns/train.py
```
