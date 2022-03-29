from estimation.graph_model import BNPGraphModel
from tricks.tricks_comb import TricksComb


class EEGNN(TricksComb):

    def __init__(self, args):
        super().__init__(args)

        self.model = None

        self.hyper_rho = 0.8

        self.alpha = 1.0
        self.tau = 1.0
        self.gamma = 1.0
        self.sigma = 0.5

        self.initial_K = 10

        self.max_K = 10


    def forward(self, x, edge_index):
        print(edge_index)

        #if self.model is None:
        #    graph = torch.sparse_coo_tensor(data.edge_index, torch.ones(number_of_edges), [number_of_nodes, number_of_nodes])
        #    self.model = BNPGraphModel(graph, self.alpha, self.tau, self.gamma, self.sigma, self.initial_K, self.max_K)


        #edge_index =

        super().forward(x, edge_index)
