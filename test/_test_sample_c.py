import torch
from torch.distributions import Gamma

from estimation.sample_c import compute_c


i = [[0, 1, 1, 2],
    [2, 0, 2, 1]]
v_z = [3, 4, 5, 2]

z = torch.sparse_coo_tensor(i, v_z, (3, 3))

max_K = 10
node_number = 4
initial_K = 2
w = Gamma(concentration=1., rate=1.).sample([max_K, node_number]) * 5.0

pi = torch.cat((torch.ones(initial_K) / initial_K, torch.zeros(max_K - initial_K)), 0)

c = compute_c(pi, torch.log(w), z)

print(c)