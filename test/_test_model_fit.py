import torch

from estimation.graph_model import BNPGraphModel

graph = [[0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]]


graph = torch.tensor(graph).to_sparse(2)
print(graph)

alpha = 0.1
tau = 1.0
gamma = 5.0
sigma = 0.5


#### test 0
model = BNPGraphModel(graph, alpha, tau, gamma, sigma, initial_K=3, max_K=100)
#### test 8
model.fit(10000)