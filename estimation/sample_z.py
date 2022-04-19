"""
this script contains the function to compute z from sparse v , pi and w
"""
import torch

from estimation.truncated_poisson import TruncatedPoisson


def compute_z(log_w: torch.tensor, pi: torch.sparse, c: torch.sparse):
    """
    This function computes the class indicators given cluster proportion vector pi and weight matrix w.
    :param c: a sparse matrix to indicate the cluster membership.
    :param log_w: weight matrix for all nodes in clusters.
    :return: a sparse matrix for the number of hidden edges
    """
    indices = c._indices()

    indices_0, indices_1 = indices[0], indices[1]
    poisson_para_tmp = torch.index_select(log_w, 1, indices_0) + torch.index_select(log_w, 1, indices_1)   # K(the number of clusters) X number of edges
    poisson_para = torch.gather(poisson_para_tmp, dim=0, index=torch.unsqueeze(c._values(), 0))   # https://zhuanlan.zhihu.com/p/352877584
    poisson_para *= torch.index_select(pi, dim=0, index=c._values())
    poisson_para = torch.where(indices_0==indices_1, poisson_para, poisson_para *2.0)
    samples = TruncatedPoisson(torch.squeeze(torch.exp(poisson_para) + 1e-10)).sample()

    z = torch.sparse_coo_tensor(indices, samples, c.size())

    return z


