"""
This script is used to add a new cluster.
"""
import torch

from estimation.stirling_number import stirling_number


def build_injection(count, active_k, max_k, gamma):
    #print('count', count)
    with_sample_indices = count > 0

    remaining_indices = torch.squeeze(torch.cat([torch.tensor([True]), with_sample_indices[1::]], dim=0).nonzero())

    dict_tmp = {}
    index = 0
    for j in range(1, active_k):
        if with_sample_indices[j]:
            index += 1
            dict_tmp[j] = index

    old_active_K = index + 1
    add_number = stirling_number(count[0], gamma)

    if old_active_K + add_number <= max_k:
        new_active_K = old_active_K + add_number
    else:
        new_active_K = max_k

    def fn(x):
        if x == 0:
            return torch.randint(low=old_active_K, high=new_active_K, size=()).item()
        else:
            return dict_tmp[x]

    return fn, new_active_K, remaining_indices


def add_k(c, active_k, max_k, gamma):
    """
    replace the cluster indictors of 0 to active_K+1
    :param c: a sparse matrix to indicate the cluster membership.
    :return: a new sparse matrix to indicate the cluster membership.
    """
    indices = c._indices()
    values = c._values()

    values_one_hot = torch.nn.functional.one_hot(values, num_classes=active_k)
    count = torch.sum(values_one_hot, dim=0)

    fn, new_active_K, remaining_indices = build_injection(count, active_k, max_k, gamma)

    values = values.apply_(fn)

    c = torch.sparse_coo_tensor(indices, values, c.size())

    return c, new_active_K, remaining_indices


def switch(inputs, remaining_indices, max_k):
    remaining = torch.index_select(inputs, dim=0, index=remaining_indices)
    deleting_indices = generating_deleting_indices(max_k, remaining_indices)
    deleting = torch.index_select(inputs, dim=0, index=deleting_indices)
    outputs = torch.cat([remaining, deleting], dim=0)
    return outputs


def generating_deleting_indices(max_k, remaining_indices):
    deleting_indices = torch.tensor([int(item) for item in torch.arange(0, max_k) if item not in remaining_indices])
    return deleting_indices


if __name__ == "__main__":

    i = [[0, 1, 1, 2],
        [2, 0, 2, 1]]
    v_c = [0, 1, 2, 0]
    active_K = 3
    c = torch.sparse_coo_tensor(i, v_c, (3, 3))

    c_new = add_k(c, active_K, max_k=10, gamma=1)
    print(c_new)