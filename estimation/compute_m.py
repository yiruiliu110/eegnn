"""
this script contains the function to compute m from sparse n and c
"""
import torch
from torch.nn.functional import one_hot


def compute_m(z: torch.sparse, c: torch.sparse, max_K: int):
    """
    This function computes the m_i = sum{n_ij}.

    :param z: the sparse matrix recoding z_ij. node_number X node_number. Each node has at least one edge required.
    :param c: the sparse matrix recoding c_ij. node_number X node_number
    :param max_K: the max of the number of clusters
    :return: m: a tensor max_K X node_number
    """

    values = torch.einsum('b,bj->bj', z._values(), one_hot(c._values()))
    size = (z.size()[0], z.size()[0], values.size()[-1])
    one_hot_m = torch.sparse_coo_tensor(z._indices(), values, size)

    values = torch.sparse.sum(one_hot_m, [0]) + torch.sparse.sum(one_hot_m, [1])   # node_number X active_K

    m = torch.transpose(values.to_dense(), 0, 1)  # active_K X node_number
    m = torch.cat([m, torch.zeros(max_K - m.size()[0], m.size()[1])])
    return m
