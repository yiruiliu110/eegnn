"""
This script is used to add a new cluster.
"""
import torch


def add_k(c, active_K):
    """
    replace the cluster indictors of 0 to active_K+1
    :param c: a sparse matrix to indicate the cluster membership.
    :return: a new sparse matrix to indicate the cluster membership.
    """
    indices = c._indices()

    values = torch.where(c._values() == 0, active_K, c._values())

    c = torch.sparse_coo_tensor(indices, values, c.size())
    return c


if __name__ == "__main__":

    i = [[0, 1, 1, 2],
        [2, 0, 2, 1]]
    v_c = [0, 1, 2, 0]
    active_K = 3
    c = torch.sparse_coo_tensor(i, v_c, (3, 3))

    c_new = add_k(c, active_K)
    print(c_new)