"""
This script tests the sampling of truncated Poisson variable.
"""

import torch

from estimation.truncated_poisson import TruncatedPoisson

rate = torch.Tensor([0.1, 0.6, 0.7, 5])

dist = TruncatedPoisson(rate)

sample = dist.sample()

print(sample)