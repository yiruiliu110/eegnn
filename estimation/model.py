from torch import nn

from estimation.eegcn import EEGCN
from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel
import torch
import torch.nn.functional as F


class EEGCNModel(torch.nn.Module):
    def __init__(self, data, number_layer=24, out_channels=16, training_epochs=1000, sampling_number=1000,
                 alpha=1.0, tau=1.0, gamma=1.0, sigma=0.5, initial_K=80, max_K=150,
                 ):
        super().__init__()
        self.number_layer = number_layer
        self.out_channels = out_channels

        x, edge_index = data.x, data.edge_index
        with torch.no_grad():
            number_of_edges = int(edge_index.size()[1])
            number_of_nodes = int(torch.max(edge_index).item()) + 1
            print(edge_index, torch.ones(number_of_edges),
                                            [number_of_nodes, number_of_nodes])
            graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                            [number_of_nodes, number_of_nodes])


            estimated_graph = BNPGraphModel(graph, alpha, tau, gamma, sigma, initial_K, max_K)
            estimated_graph.fit(training_epochs)
            mean_pi, mean_log_w = estimated_graph.compute_mean(sampling_number)
            global_edge_index, fixed_edge_weight = compute_dege_index_and_weight(mean_pi, mean_log_w[:, 0:-1])

        conv_layers = []
        for i in range(0, number_layer):
            if i == 0:
                start, end = int(data.num_node_features), out_channels
            elif i == number_layer - 1:
                start, end = out_channels, int(data.y.max()) + 1
            else:
                start, end = out_channels, out_channels

            layer = EEGCN(in_channels=start, out_channels=end,
                          global_edge_index=global_edge_index,
                          fixed_edge_weight=fixed_edge_weight,
                          fixed_feature=x)
            conv_layers.append(layer)

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.number_layer - 1):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x,  training=self.training)

        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)