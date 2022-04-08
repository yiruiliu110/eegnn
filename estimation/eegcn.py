import torch
from torch import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor

from estimation.graph_model import BNPGraphModel


class EEGCNConv(GCNConv):

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, hyper_proportion=0.4, **kwargs):
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
        torch.set_printoptions(profile="full")

        if self.graph is None:
            number_of_edges = int(edge_index.size()[1])
            number_of_nodes = int(torch.max(edge_index).item()) + 1
            #print('test', number_of_nodes)
            self.graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                            [number_of_nodes, number_of_nodes])

        if self.estimated_graph is None:
            self.estimated_graph = BNPGraphModel(self.graph, alpha=10.0, tau=1.0, gamma=1.0, sigma=0.5, initial_K=50, max_K=100)
            self.estimated_graph.fit(1000)
        #print("AAAAAAA")

        edge_index_sample = self.estimated_graph.sample()
        #print('samplet', torch.max(edge_index_sample))

        output_1 = self.one_step(x, edge_index_sample, edge_weight)

        output_0 = self.one_step(x, edge_index, edge_weight)
        return output_0 * (1. - self.hyper_proportion) + self.hyper_proportion * output_1

        #number_of_edges = self.estimated_graph.edge_number
        #edge_index = edge_index[:, torch.randperm(number_of_edges)]#[0:int(number_of_edges * (1. - self.hyper_proportion))]]
        #edge_index_sample = edge_index_sample[:, torch.randperm(self.estimated_graph.total_number_sampled_edges)[0:int(number_of_edges * self.hyper_proportion)]]
        #print(edge_index_sample)
        #edge_index = torch.cat([edge_index, edge_index_sample], dim=1)
        #return self.one_step(x, edge_index, edge_weight)

    def one_step(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out




