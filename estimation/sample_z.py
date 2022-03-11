"""
this script contains the function to compute z from sparse v , pi and w
"""
import torch
from torch.distributions import Categorical

from estimation.truncated_poisson import TruncatedPoisson


def compute_z(w: torch.tensor, c: torch.sparse, graph: torch.sparse):
    """
    This function computes the class indicators given cluster proportion vector pi and weight matrix w.
    :param c: a sparse matrix to indicate the cluster membership.
    :param w: weight matrix for all nodes in clusters.
    :param graph: the sparse matrix recoding the observed graph.
    :return: a sparse matrix for the number of hidden edges
    """
    indices = graph._indices()

    indices_0, indices_1 = indices[0], indices[1]

    poisson_para_tmp = torch.index_select(w, 1, indices_0) * torch.index_select(w, 1, indices_1)   # K(the number of clusters) X number of edges
    poisson_para = torch.gather(poisson_para_tmp, dim=0, index=torch.unsqueeze(c._values(), 0))   # https://zhuanlan.zhihu.com/p/352877584

    samples = TruncatedPoisson(torch.squeeze(poisson_para)).sample()

    z = torch.sparse_coo_tensor(indices, samples, c.size())

    return z


