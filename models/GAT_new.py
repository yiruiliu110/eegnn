import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from models.initial_graph import initial_graph


class GAT_new(torch.nn.Module):
    def __init__(self, args):
        super(GAT_new, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])

        # space limit
        if self.dataset == 'obgn-arxiv':
            self.dim_hidden = 1

        self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden,
                                       bias=False, concat=False))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GATConv(self.dim_hidden, self.dim_hidden, bias=False, concat=False, add_self_loops=False))
            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, bias=False, concat=False, add_self_loops=False))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.virtual_graph = None

    def forward(self, x, edge_index):

        if self.virtual_graph is None:
            self.virtual_graph = initial_graph(edge_index, self.dataset)
            self.virtual_edge_index = self.virtual_graph._indices()
            self.virtual_edge_weight = self.virtual_graph._values()


        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm == 'batch':
                x = self.layers_bn[i](x)
            x = F.relu(x)

        x = self.layers_GCN[-1](x, edge_index)
        return x
