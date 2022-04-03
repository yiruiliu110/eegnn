import torch
from torch import Tensor

from torch_geometric.nn import GCNConv

from torch_geometric.typing import Adj, OptTensor, PairTensor

from estimation.graph_model import BNPGraphModel


class EEGCNConv(GCNConv):

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, hyper_proportion=0.5, **kwargs):
        super(EEGCNConv, self).__init__(in_channels, out_channels,
                                    improved, cached,
                                    add_self_loops, normalize,
                                    bias, **kwargs)
        self.graph = None

        self.estimated_graph = None

        self.hyper_proportion = hyper_proportion

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        #edge_index: torch.Tensor 2 X number_of_edges

        if self.graph is None:
            number_of_edges = int(edge_index.size()[1])
            number_of_nodes = int(torch.max(edge_index).item()) + 1
            self.graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                            [number_of_nodes, number_of_nodes])

        if self.estimated_graph is None:
            self.estimated_graph = BNPGraphModel(self.graph, alpha=10.0, tau=1.0, gamma=5.0, sigma=0.5, initial_K=3, max_K=20)
            self.estimated_graph.fit(1000)

        edge_index_sample = self.estimated_graph.sample()

        output_1 = super(EEGCNConv, self).forward(x, edge_index_sample, edge_weight)

        output_0 = super(EEGCNConv, self).forward(x, edge_index, edge_weight)
        return output_0 * (1. - self.hyper_proportion) + self.hyper_proportion * output_1


