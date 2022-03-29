import torch
from torch.distributions import Dirichlet


def sample_pi(n, alpha):
    parameter = torch.cat([torch.Tensor([alpha]), n+alpha], dim=0) + 1e-4
    return Dirichlet(parameter).sample()

