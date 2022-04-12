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
                 bias: bool = True,  **kwargs):
        super(EEGCNConv, self).__init__(in_channels, out_channels,
                                    improved, cached,
                                    add_self_loops, normalize,
                                    bias, **kwargs)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        output_1 = self.one_step(x, edge_index, edge_weight)
        return output_1

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




