
import torch

def compute_dege_index_and_weight(mean_pi, mean_log_w):
    """

    :param mean_pi: K,
    :param mean_log_w: K X N_node
    :return: 2 X N-edge, N_edge
    """
    mean_w = torch.exp(mean_log_w)
    mat = torch.transpose(mean_w, 0, 1) @ torch.diag(mean_pi) @ mean_w
    mat = mat.type(torch.float64)

    mat_sum = torch.sum(mat, dim=0)
    mat_sqrt = torch.nan_to_num(1.0 / torch.sqrt(mat_sum))

    mat = (mat) * (torch.unsqueeze(mat_sqrt, 1) @ torch.unsqueeze(mat_sqrt, 0))
    mat = torch.where(mat > 1e-4, mat, 0.0).type(torch.float32)

    mat_sparse = mat.to_sparse()
    #print(mat_sparse._indices().size())

    #index = mat_sparse._indices()
    #weight = mat_sparse._values()

    return mat_sparse #index, weight



