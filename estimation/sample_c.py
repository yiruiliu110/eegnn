"""
this script contains the function to compute c from sparse v , pi and w
"""
import torch
from torch.distributions import Categorical


def compute_c(pi: torch.Tensor, log_w: torch.tensor, z: torch.sparse):
    """
    This function computes the class indicators given cluster proportion vector pi and weight matrix w.

    :param pi: the cluster proportion
    :param log_w: weight matrix for all nodes in clusters.
    :param z: the sparse matrix recoding n_ij. node_number X node_number. Each node has at least one edge required.
    :return: a sparse matrix to indicate the cluster membership.
    """

    # construct a N X active_K matrix, N is the number of observed edges
    indices = z._indices()
    indices_0, indices_1 = indices[0], indices[1]
    weight = (torch.index_select(log_w, 1, indices_0) + torch.index_select(log_w, 1, indices_1)) * 2.0\
             + torch.log(torch.unsqueeze(pi, dim=1) + 1e-4)

    c_tmp = Categorical(logits=torch.transpose(weight, 0, 1)).sample()

    c = torch.sparse_coo_tensor(indices, c_tmp, z.size())
    #print('c', c._values()[0:25])
    return c
