import math
import pickle

import torch
import torch.nn.functional as F
from torch import nn

from models.GCNII_DenseLayer import GCNIIConv_arxiv
from models.GCNII_layer import GCNIIdenseConv

from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel
from models.initial_graph import initial_graph


class GCNII_new(nn.Module):
    def __init__(self, args):
        super(GCNII_new, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive

        gcn_conv = GCNIIConv_arxiv if self.dataset == 'ogbn-arxiv' else GCNIIdenseConv

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers):
            self.convs.append(gcn_conv(self.dim_hidden, self.dim_hidden, add_self_loops=False))
        self.convs.append(torch.nn.Linear(self.dim_hidden, self.num_classes))

        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())

        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

        self.virtual_graph = None

    def forward(self, x, edge_index):

        if self.virtual_graph is None:
            self.virtual_graph = initial_graph(edge_index)
            self.virtual_edge_index = self.virtual_graph._indices()
            self.virtual_edge_weight = self.virtual_graph._values()

        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[0](x)
        x = F.relu(x)
        x_init = x
        x_last = x

        for i, con in enumerate(self.convs[1:-1]):
            if self.dataset != 'ogbn-arxiv':
                beta = math.log(self.lamda / (i + 1) + 1)

                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(con(x, self.virtual_edge_index, self.alpha, x_init, beta=beta,
                               edge_weight=self.virtual_edge_weight, ))


            else:
                x = F.relu(con(x, edge_index, self.alpha, x_init)) + x_last
                x_last = x

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)

        return x

