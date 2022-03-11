import torch
from torch.distributions import Gamma

from estimation.sample_z import compute_z

i = [[0, 1, 1, 2],
    [2, 0, 2, 1]]
v_graph = [1, 1, 1, 1]
v_c = [0, 1, 1, 0]

graph = torch.sparse_coo_tensor(i, v_graph, (3, 3))
c = torch.sparse_coo_tensor(i, v_c, (3, 3))

max_K = 10
node_number = 4
w = Gamma(concentration=1., rate=1.).sample([max_K, node_number]) * 5.0

z = compute_z(w, c, graph)

print(z)