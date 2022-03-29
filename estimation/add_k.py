"""
This script is used to add a new cluster.
"""
import torch


def build_injection(count, active_K, max_k):
    with_sample_indices = count > 0
    without_sample_indices = count <= 0

    remaining_indices = torch.squeeze(torch.cat([torch.tensor([True]), with_sample_indices[1::]], dim=0).nonzero())
    deleting_indices = torch.squeeze(torch.cat([torch.tensor([False]), without_sample_indices[1::]], dim=0).nonzero())

    dict_tmp = {}
    index = 1
    for j in range(1, active_K):
        if with_sample_indices[j]:
            dict_tmp[j] = index
            index += 1

    new_active_K = len(dict_tmp) + 1
    if new_active_K + 1 <= max_k:
        dict_tmp[0] = new_active_K
        new_active_K += 1
    else:
        dict_tmp[0] = torch.randint(low=1, high=max_k, size=()).item()

    def fn(x):
        return dict_tmp[x]

    return fn, new_active_K, remaining_indices, deleting_indices


def add_k(c, pi, log_w, active_K, max_k):
    """
    replace the cluster indictors of 0 to active_K+1
    :param c: a sparse matrix to indicate the cluster membership.
    :return: a new sparse matrix to indicate the cluster membership.
    """
    indices = c._indices()
    values = c._values()

    values_one_hot = torch.nn.functional.one_hot(values, num_classes=active_K)
    count = torch.sum(values_one_hot, dim=0)

    fn, new_active_K, remaining_indices, deleting_indices = build_injection(count, active_K, max_k)

    values = values.apply_(fn)  #TODO active_K-1, random_index

    c = torch.sparse_coo_tensor(indices, values, c.size())
    pi_remaining = torch.index_select(pi, dim=0, index=remaining_indices)
    pi = torch.cat([pi_remaining, torch.zeros(max_k - new_active_K)], dim=0)
    log_w_remaining = torch.index_select(log_w, dim=0, index=remaining_indices)
    log_w_deleting = torch.index_select(log_w, dim=0, index=deleting_indices)
    log_w = torch.cat([log_w_remaining, log_w_deleting], dim=0)

    return c, pi, log_w, new_active_K


if __name__ == "__main__":

    i = [[0, 1, 1, 2],
        [2, 0, 2, 1]]
    v_c = [0, 1, 2, 0]
    active_K = 3
    c = torch.sparse_coo_tensor(i, v_c, (3, 3))

    c_new = add_k(c, active_K)
    print(c_new)