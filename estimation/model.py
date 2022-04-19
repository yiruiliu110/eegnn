import pickle

from torch import nn

from estimation.eegcn import EEGCN
from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class EEGCNModel(torch.nn.Module):
    def __init__(self, data, number_layer=24, out_channels=64, training_epochs=100, sampling_number=100,
                 alpha=1.0, tau=1.0, gamma=1.0, sigma=0.5, initial_K=10, max_K=100,
                 ):
        super().__init__()
        self.number_layer = number_layer
        self.out_channels = out_channels

        self.alpha = Variable(torch.tensor([0.1]), requires_grad=True)
        self.gamma = Variable(torch.tensor([0.1]), requires_grad=True)

        x, edge_index = data.x, data.edge_index
        try:
            with open("global_edge_index_0", "rb") as output_file:
                self.global_edge_index = pickle.load(output_file)
            with open("fixed_edge_weight_0", "rb") as output_file:
                self.fixed_edge_weight = pickle.load(output_file)
        except:
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
                self.global_edge_index, self.fixed_edge_weight = compute_dege_index_and_weight(mean_pi, mean_log_w[:, 0:-1])
                with open("global_edge_index_0", "wb") as output_file:
                    pickle.dump(self.global_edge_index, output_file)
                with open("fixed_edge_weight_0", "wb") as output_file:
                    pickle.dump(self.fixed_edge_weight, output_file)

        conv_layers = []
        for i in range(0, number_layer):
            if i == 0:
                layer = nn.Linear(int(data.num_node_features), out_channels)
            elif i == number_layer - 1:
                layer = nn.Linear(out_channels, int(data.y.max()) + 1)
            else:
                layer = EEGCN(channels=out_channels, alpha=self.alpha, gamma=self.gamma, )
            conv_layers.append(layer)

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, training=self.training)
        x = self.layers[0](x)
        x_0 = x
        x_1 = x_0

        for i in range(1, self.number_layer - 1):
            x = F.dropout(x, training=self.training)
            if i == self.number_layer - 2:
                x = self.layers[i](x, x_0=x_0, x_1=x_1, edge_index=edge_index, edge_index_global=self.global_edge_index,
                                   edge_weight_global=self.fixed_edge_weight, )
            else:
                x = self.layers[i](x, x_0=x_0, x_1=x_1, edge_index=edge_index, edge_index_global=self.global_edge_index,
                                   edge_weight_global=self.fixed_edge_weight, )
            x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)
