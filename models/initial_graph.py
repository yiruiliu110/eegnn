import pickle

import torch

from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel


def initial_graph(edge_index):
    try:
        with open("global_edge_index_0", "rb") as output_file:
            virtual_graph = pickle.load(output_file)


    except:
        with torch.no_grad():
            number_of_edges = int(edge_index.size()[1])
            number_of_nodes = int(torch.max(edge_index).item()) + 1
            print(edge_index, torch.ones(number_of_edges),
                  [number_of_nodes, number_of_nodes])
            graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                            [number_of_nodes, number_of_nodes])

            estimated_graph = BNPGraphModel(graph, alpha=1.0, tau=1.0, gamma=1.0, sigma=0.5, initial_K=100, max_K=150)
            estimated_graph.fit(10000)
            mean_pi, mean_log_w = estimated_graph.compute_mean(1000)
            virtual_graph = compute_dege_index_and_weight(mean_pi, mean_log_w[:, 0:-1]).requires_grad_(False)
            with open("global_edge_index_0", "wb") as output_file:
                pickle.dump(virtual_graph, output_file)
    print(virtual_graph)
    return virtual_graph