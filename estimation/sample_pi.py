import torch
from torch.distributions import Dirichlet


def sample_pi(n, alpha):
    parameter = torch.cat([torch.tensor([alpha]), n[1::]], dim=0) + 1e-15
    return Dirichlet(parameter).sample()

