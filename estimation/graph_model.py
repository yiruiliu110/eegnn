"""
This file contain the class for Bayesian nonparametric graph model
"""
import functools

import torch
from torch import triu
from torch.distributions import Gamma, Dirichlet

from estimation.compute_m import compute_m
from estimation.mh import MetropolisHastings
from estimation.sample_c import compute_c
from estimation.sample_z import compute_z


class BNPGraphModel(object):

    def __init__(self, graph: torch.Tensor,
                 alpha: float,
                 tau: float,
                 gamma: float,
                 sigma: float = 0.0,
                 initial_K: int = 20,
                 max_K: int = 100):

        # B convention, we just need to infer upper triangle's n_ij
        dense_graph = triu(graph.to_dense())
        self.graph_sparse = dense_graph.to_sparse(2)

        self.node_number = dense_graph.shape[0]
        self.edge_number = torch.sum(dense_graph).to(int)
        print(f"Num Nodes : {self.node_number} \t Num Edges : {self.edge_number}")

        # number of clusters
        self.max_K = max_K
        self.active_K = initial_K

        # other settings
        self.proposal_r_idx, self.proposal_c_idx = torch.split(graph._indices(), [1, 1])  # For Step 3, pre-define indices

        # Step 0: Initialization
        indices = self.graph_sparse._indices()
        size = self.graph_sparse.size()
        self.state = {
            'pi': torch.cat([torch.ones(initial_K) / initial_K, torch.zeros(max_K - initial_K)]),
            'w_0':  Gamma(concentration=1., rate=1.).sample([self.node_number]),
            'w': Gamma(concentration=1., rate=1.).sample([max_K, self.node_number]),
            #'w_star': Gamma(concentration=1., rate=1.).sample([max_K]),
            'z': torch.sparse_coo_tensor(indices=indices,
                                         values=torch.randint(low=1, high=5, size=(self.edge_number,)),
                                         size=size),
            'c': torch.sparse_coo_tensor(indices=indices,
                                         values=torch.randint(low=0, high=initial_K, size=(self.edge_number,)),
                                         size=size),
        }

        self.state['m'] = compute_m(self.state['z'], self.state['c'], max_K)

        # hyper parameters
        self.hyper_paras = {
            'alpha': alpha,
            'gamma': gamma,  # the hyper_para for Dirichlet process
            'sigma': sigma,
            'tau': tau,
        }

        self.w_0_mh = MetropolisHastings()
        self.w_k_mh = MetropolisHastings()

    def one_step(self):
        self.update_w_0_proportion()
        self.update_w_proportion()
        self.update_pi()
        self.update_c()
        self.update_z()
        self.update_w_0_total()
        self.update_w_total()

    def update_w_proportion(self):
        self.state['m'] = compute_m(self.state['z'], self.state['c'], self.max_K)
        concentration = self.state['m'] + torch.unsqueeze(self.state['w_0'], dim=1)
        w_tmp = Gamma(concentration=concentration, rate=1.).sample()  # max_K X number_nodes

        self.state['w'] = w_tmp / torch.sum(w_tmp, dim=1, keepdim=True) * torch.sum(self.state['w'], dim=1, keepdim=True)

    def update_pi(self):
        m_tmp = torch.sum(self.state['m'], dim=1)  # the number of links in each clusters, first row corresponds to cluster 0.
        parameter = torch.cat([torch.Tensor([self.hyper_paras['alpha']]), m_tmp[1:self.active_K]], dim=0)
        pi_tmp = Dirichlet(parameter).sample()
        self.state['pi'] = torch.cat([pi_tmp, torch.zeros(self.max_K - self.active_K)], dim=0)

    def update_c(self):
        self.state['c'] = compute_c(self.state['pi'], self.state['w'], self.state['z'])

    def update_z(self):
        self.state['z'] = compute_z(self.state['w'], self.state['c'], self.graph_sparse)

    def update_w_0_total(self):
        log_prob_fn = functools.partial(self.log_prob_wrt_w_0, w_k=self.state['w_k'],
                                        u=self.u)
        self.w_0_mh.one_step(state=self.state['w_0'], log_prob_fn=log_prob_fn)

    @staticmethod
    def log_prob_wrt_w_0(w_0, w_k, u):
            return torch.log(u(w_0)) + w_0 * torch.sum(torch.log(w_k)) - torch.tensor(w_k.size()) * torch.lgamma(w_0)

    @staticmethod
    def log_prob_wrt_w_k(w_k, n_k, w_0, tau, pi_k):
            return (2.0 * n_k + w_0 - 1.0) * torch.log(w_k) - tau * w_k - w_k * w_k * pi_k

    def update_w_total(self):
        log_prob_fn = functools.partial(self.log_prob_wrt_w_k, n_k=torch.sum(self.state['m'], dim=1) , w_0=self.state['w_0'],
                                        tau=self.hyper_paras['tau'], pi_k=self.state['pi'])
        self.w_k_mh.one_step(state=self.state['w_k'], log_prob_fn=log_prob_fn)



