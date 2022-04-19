"""
This function builds up the initialization parameters.
"""
import torch

from estimation.compute_m import compute_m


def compute_n(m):
    return torch.sum(m, dim=1)  # the number of links in each clusters, first row corresponds to cluster 0.


def build_initials(initial_K, max_K, graph_sparse, node_number, edge_number):
    indices = graph_sparse._indices()
    size = graph_sparse.size()

    state = {
        'pi': torch.cat([torch.ones(initial_K) / initial_K, torch.zeros(max_K - initial_K)]),  # max_K includes the cluster 0.
        'log_w_0': torch.zeros(node_number + 1) - torch.log(torch.Tensor([node_number+1])),  # index -1 is the unobserved node

        'log_w': torch.zeros(max_K, node_number + 1) - torch.log(torch.Tensor([node_number+1])),
        'z': torch.sparse_coo_tensor(indices=indices,
                                     values=torch.randint(low=1, high=2, size=(edge_number,)),
                                     size=size),
        'c': torch.sparse_coo_tensor(indices=indices,
                                     values=torch.randint(low=1, high=initial_K, size=(edge_number,)), # initial_K includes the cluster 0.
                                     size=size),
    }
    state['log_w_0_total'] = torch.log(torch.sum(torch.exp(state['log_w_0'])))
    state['log_w_total'] = torch.log(torch.sum(torch.exp(state['log_w']), dim=1))

    state['m'] = compute_m(state['z'], state['c'], max_K)
    state['n'] = compute_n(state['m'])   # n_k

    return state