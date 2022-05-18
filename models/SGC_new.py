import torch
from torch import nn

# from torch_geometric.nn import SGConv
from torch_sparse import SparseTensor

from estimation.sparse_to_dense import scipy_to_dense
from models.SGC_layer import SGConv
from models.initial_graph import initial_graph


class SGC_new(nn.Module):
    def __init__(self, args):
        super(SGC_new, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        # bn used in dataset 'obgn-arxiv'
        # lin_first used in Coauthor_physics due to space limit
        self.bn = True if args.type_norm == 'batch' else False
        self.lin_first = True if args.dataset == 'Coauthor_Physics' else False
        self.SGC = SGConv(self.num_feats, self.num_classes, K=self.num_layers,
                          cached=self.cached, bias=False, bn=self.bn, dropout=self.dropout, add_self_loops=False,
                          lin_first=self.lin_first)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.virtual_graph=None


    def forward(self, x, edge_index):



        if self.virtual_graph is None:
            self.virtual_graph = initial_graph(edge_index, self.dataset)

        edge_index, edge_weight = self.virtual_graph.gnn_sample()


        # implemented based on https://github.com/Tiiiger/SGC/blob/master/citation.py
        x = self.SGC(x, edge_index=edge_index, edge_weight=edge_weight)
        return x








