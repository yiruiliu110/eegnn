import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from estimation.sparse_to_dense import scipy_to_dense
from models.initial_graph import initial_graph


class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class GCN_new(nn.Module):
    def __init__(self, args):
        super(GCN_new, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])

        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, add_self_loops=False))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached, add_self_loops=False))

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())
        self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached))

        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.virtual_graph=None

    def forward(self, x, edge_index):

        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py
        if self.virtual_graph is None:
            self.virtual_graph = initial_graph(edge_index, self.dataset)

        if isinstance(self.virtual_graph, torch.Tensor):
            edge_index, edge_weight = scipy_to_dense(self.virtual_graph)

        elif isinstance(edge_index, SparseTensor):
            edge_index = self.virtual_graph._indices()
            edge_weight = self.virtual_graph._values()

        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index=edge_index, edge_weight=edge_weight)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)
        return x
