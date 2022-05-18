import torch
def scipy_to_dense(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    #A = A.tocoo()
    indices = A._indices()
    row = indices[0].to(torch.long)
    col = indices[1].to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = A._values()
    return edge_index, edge_weight