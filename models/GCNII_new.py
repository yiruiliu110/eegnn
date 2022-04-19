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
            self.convs.append(gcn_conv(self.dim_hidden, self.dim_hidden))
        self.convs.append(torch.nn.Linear(self.dim_hidden, self.num_classes))

        # new
        #self.convs_new = torch.nn.ModuleList()
        #self.convs_new.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
        #for _ in range(self.num_layers):
        #    self.convs_new.append(gcn_conv(self.dim_hidden, self.dim_hidden, cached=False))
        #self.convs_new.append(torch.nn.Linear(self.dim_hidden, self.num_classes))

        # interface
        #self.interfaces = torch.nn.Linear(self.dim_hidden * 2, self.dim_hidden)

        self.reg_params = list(self.convs[1:-1].parameters()) #+ list(self.convs_new[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters()) #+ \
                              #list(self.convs_new[0:1].parameters()) + list(self.convs_new[-1:].parameters())  + \
                              #list(self.interfaces.parameters())

        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

        self.virtual_graph = None


    def forward(self, x, edge_index):

        if self.virtual_graph is None:
            self.virtual_graph = initial_graph(edge_index)

        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[0](x)
        x = F.relu(x)
        x_init = x
        x_last = x

        #x_1 = F.dropout(x_1, self.dropout, training=self.training)
        #x_1 = self.convs_new[0](x_1)
        #x_1 = F.relu(x_1)
        #x_1_init = x_1
        #x_1_last = x_1

        #for i, (con, con_new) in enumerate(zip(self.convs[1:-1], self.convs_new[1:-1])):
        for i, con in enumerate(self.convs[1:-1]):


            if self.dataset != 'ogbn-arxiv':
                beta = math.log(self.lamda / (i + 1) + 1)

                x_1_out = torch.sparse.mm(self.virtual_graph, x_init)
                x_1_out = (1-self.alpha) * x_1_out +  self.alpha * x_init
                x = F.dropout(x, self.dropout, training=self.training)
                print('A', torch.max(x_1_out))
                x = F.relu(con(x, edge_index, self.alpha, x_init, beta)) #x_ini
                print('B', torch.max(x))
                x = self.merge_inputs(x, x_1_out)


            else:
                x = F.relu(con(x, edge_index, self.alpha, x_init)) + x_last
                x_last = x

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        #x_1 = F.dropout(x_1, self.dropout, training=self.training)
        #x_1 = self.convs_new[-1](x_1)

        out = x#self.merge_inputs(x, x_1)
        return out

    def merge_inputs(self, x_0, x_1):
        z = 0.7
        one_minus_z = 1.0 - z

        #return z*x_0 + one_minus_z * x_1

        #return x_1

        return torch.maximum(x_0, x_1)
        #return x_0 +x_1
