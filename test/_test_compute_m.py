import torch

from estimation import compute_m

i = [[0, 1, 1, 2],
    [2, 0, 2, 1]]
v_z = [3, 4, 5, 2]
v_c = [0, 1, 1, 0]

z = torch.sparse_coo_tensor(i, v_z, (3, 3))
c = torch.sparse_coo_tensor(i, v_c, (3, 3))

max_K = 10

m = compute_m(z, c, max_K)

print(m)