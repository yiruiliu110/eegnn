from abc import ABC

import torch
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor


class EEGCN(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, global_edge_index, fixed_edge_weight, fixed_feature,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True,  **kwargs):
        super(EEGCN, self).__init__(**kwargs)

        self.global_layer = GlobalConv(in_channels, out_channels, global_edge_index, fixed_edge_weight,
                 improved, cached, add_self_loops, normalize, bias,  **kwargs)

        fixed_in_channels = int(fixed_feature.size()[1])

        self.local_layer = LocalConv(fixed_in_channels, out_channels, fixed_feature,
                 improved, cached, add_self_loops, normalize, bias,  **kwargs)

        self.lin_total = Linear(2 * out_channels, out_channels, bias=False, weight_initializer='glorot')

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        global_out = self.global_layer.forward(x, edge_index, edge_weight)
        local_out = self.local_layer.forward(x, edge_index, edge_weight)

        out = self.lin_total(torch.cat([global_out, local_out], dim=1))

        return out


class GlobalConv(GCNConv):

    def __init__(self, in_channels: int, out_channels: int, fixed_edge_index, fixed_edge_weight,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):
        super(GlobalConv, self).__init__(in_channels, out_channels,
                                         improved, cached,
                                         add_self_loops, normalize,
                                         bias, **kwargs)
        self.global_edge_index = fixed_edge_index
        self.global_edge_weight = fixed_edge_weight

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        out = super(GlobalConv, self).forward(x, self.global_edge_index, self.global_edge_weight)
        return out


class LocalConv(GCNConv):

    def __init__(self, in_channels: int, out_channels: int, fixed_feature,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True,  **kwargs):
        super(LocalConv, self).__init__(in_channels, out_channels,
                                        improved, cached,
                                        add_self_loops, normalize,
                                        bias, **kwargs)
        self.fixed_feature = fixed_feature

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        out = super(LocalConv, self).forward(self.fixed_feature, edge_index, edge_weight)
        return out




