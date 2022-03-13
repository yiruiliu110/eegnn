"""
This file contain the class for Bayesian nonparametric graph model
"""
import functools

import torch
from torch.distributions import Gamma, Dirichlet

from estimation.build_initials import build_initials
from estimation.compute_m import compute_m
from estimation.hmc import HamiltonMonteCarlo
from estimation.mh import MetropolisHastings
from estimation.sample_c import compute_c
from estimation.sample_pi import sample_pi
from estimation.sample_w_proportion import sample_w_proportion
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
        dense_graph = torch.triu(graph.to_dense())
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
        self.state = build_initials(initial_K, max_K, self.graph_sparse, self.node_number, self.edge_number)

        # hyper parameters
        self.hyper_paras = {
            'alpha': alpha,
            'gamma': gamma,  # the hyper_para for Dirichlet process
            'sigma': sigma,
            'tau': tau,
        }

        self.w_0_mh_sampler = MetropolisHastings()
        self.w_k_mh_sampler = MetropolisHastings()
        self.w_0_proportion_sampler = HamiltonMonteCarlo()

    def one_step(self):
        self.update_w_0_proportion()
        self.update_w_proportion()
        self.update_pi()
        self.update_c()
        self.update_z()
        self.update_w_0_total()
        self.update_w_total()

    def update_w_proportion(self):
        w_bar = torch.sum(torch.exp(self.state['log_w'][1:self.active_K]), dim=1)
        log_w = sample_w_proportion(self.state['m'][1:self.active_K], self.state['log_w_0'], w_bar)
        self.state['log_w'] = torch.cat([self.state['log_w'][0:1], log_w, self.state['log_w'][self.active_K::]], dim=0)

    def update_pi(self):
        # the number of links in each clusters, first row corresponds to cluster 0.
        pi = sample_pi(self.state['n'][1:self.active_K], self.hyper_paras['alpha'])
        self.state['pi'] = torch.cat([pi, torch.zeros(self.max_K - self.active_K)], dim=0)

    def update_c(self):
         c = compute_c(self.state['pi'][0:self.active_K], self.state['log_w'][0:self.active_K], self.state['z'])
         self.state['c'] = c

    def update_z(self):
        self.state['z'] = compute_z(self.state['w'], self.state['c'], self.graph_sparse)

    def update_w_0_total(self):
        log_prob_fn = functools.partial(self.log_prob_wrt_w_0, w_k=self.state['w_k'],
                                        u=self.u)
        self.w_0_mh_sampler.one_step(state=self.state['w_0'], log_prob_fn=log_prob_fn)

    @staticmethod
    def log_prob_wrt_w_0(w_0, w_k, u):
            return torch.log(u(w_0)) + w_0 * torch.sum(torch.log(w_k)) - torch.tensor(w_k.size()) * torch.lgamma(w_0)

    @staticmethod
    def log_prob_wrt_w_k(w_k, n_k, w_0, tau, pi_k):
            return (2.0 * n_k + w_0 - 1.0) * torch.log(w_k) - tau * w_k - w_k * w_k * pi_k

    def update_w_total(self):
        log_prob_fn = functools.partial(self.log_prob_wrt_w_k, n_k=torch.sum(self.state['m'], dim=1), w_0=self.state['w_0'],
                                        tau=self.hyper_paras['tau'], pi_k=self.state['pi'])
        self.w_k_mh_sampler.one_step(state=self.state['w_k'], log_prob_fn=log_prob_fn)

    def update_w_0_proportion(self):
        pass

    @staticmethod
    def log_prob_wrt_w_0_bar(n_k, w_0, u, v):
        return torch.sum(torch.lgamma(n_k + w_0) - torch.lgamma(w_0)) + torch.sum(torch.log(v(w_0)))  # TODO

