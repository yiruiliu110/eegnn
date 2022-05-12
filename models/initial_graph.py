import os
import pickle

import torch

from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel


def initial_graph(edge_index, data_name: str = 'TEXAS'):
    with torch.no_grad():
        number_of_edges = int(edge_index.size()[1])
        number_of_nodes = int(torch.max(edge_index).item()) + 1

        graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                        [number_of_nodes, number_of_nodes])

        estimated_graph = BNPGraphModel(graph, alpha=1.0, tau=1.0, gamma=5.0, sigma=0.5, initial_K=10, max_K=100)

    try:
        with open(os.path.join('data', data_name, "trained_model_state_" + data_name), "rb") as output_file:
            estimated_graph.state = pickle.load(output_file)
            estimated_graph.active_K = estimated_graph.state['active_K']

    except:
            estimated_graph.fit(10000)
            with open(os.path.join('data', data_name, "trained_model_state_" + data_name), "wb") as output_file:
                pickle.dump(estimated_graph.state, output_file)

    virtual_graph = estimated_graph.compute_mean_z(100)

    virtual_graph = virtual_graph.to_dense()
    virtual_graph = (virtual_graph + torch.transpose(virtual_graph, 0, 1)) / 2
    virtual_graph = virtual_graph.to_sparse()

    return virtual_graph