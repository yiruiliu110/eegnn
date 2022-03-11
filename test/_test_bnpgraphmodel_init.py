"""
This script tests the init function of BNPGraphModel.
"""
import torch

from estimation.graph_model import BNPGraphModel

graph = [[0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]]


graph = torch.tensor(graph).to_sparse(2)
print(graph)

alpha = 300.
tau = 1.0
gamma = 5.0
sigma = 0.5

model = BNPGraphModel(graph, alpha, tau, gamma, sigma, initial_K=20, max_K=100)


print(model.state)