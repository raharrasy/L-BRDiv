from itertools import chain
import torch.nn as nn

def fc_network(layer_dims, init_ortho=True, with_relu=False):
    """
    Builds a fully-connected NN with ReLU activation functions.

    """
    if init_ortho: 
        init = init_orthogonal
    else:
        init = lambda m: m

    network = nn.Sequential(
                *chain(
                    *((init(nn.Linear(layer_dims[i], layer_dims[i+1])),
                       nn.ReLU())
                      for i in range(len(layer_dims)-1))
                    ),
                )

    if not with_relu:
        del network[-1]  # remove the final ReLU layer
    return network

class LSTMNetwork(nn.Module):
    """
    An implementation of networks used by the AHT agent. It has LSTM layers at the end.
    """
    def __init__(self, layer_dims, init_ortho=True, with_relu=False, device="cpu"):
        super(LSTMNetwork, self).__init__()
        self.layer_dims = layer_dims
        self.device = device

        self.fc_network = fc_network(layer_dims[:-1], init_ortho, with_relu=False).to(self.device)
        self.rep_gen = nn.LSTM(layer_dims[-2], layer_dims[-1], batch_first=False).to(self.device)
        self.with_relu = with_relu

    def forward(self, input, input_c1, input_c2):
        n_out = self.fc_network(input.to(self.device))
        n_out, input_c1, input_c2 = n_out.unsqueeze(0), input_c1.unsqueeze(0), input_c2.unsqueeze(0)
        ag_reps, new_c = self.rep_gen(n_out, (input_c1, input_c2))

        return ag_reps.squeeze(0), new_c[0].squeeze(0), new_c[1].squeeze(0)

def init_orthogonal(m):
    nn.init.orthogonal_(m.weight)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)
    return m
