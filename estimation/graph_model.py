"""
This file contain the class for Bayesian nonparametric graph model
"""
import functools

import torch
from torch.distributions import Categorical

from estimation.add_k import add_k
from estimation.build_initials import build_initials, compute_n
from estimation.compute_m import compute_m
from estimation.hmc import HamiltonMonteCarlo
from estimation.log_probs import log_prob_wrt_w_0_total, log_prob_wrt_w_k_total, log_prob_wrt_w_0_proportional, \
    d_log_prob_wrt_w_0_proportional
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
                 max_K: int = 100,
                 cmr: str = 'gamma'):
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
        self.proposal_r_idx, self.proposal_c_idx = torch.split(graph._indices(),
                                                               [1, 1])  # For Step 3, pre-define indices

        # Step 0: Initialization
        self.state = build_initials(initial_K, max_K, self.graph_sparse, self.node_number, self.edge_number)

        # hyper parameters
        self.hyper_paras = {
            'alpha': alpha,
            'gamma': gamma,  # the hyper_para for Dirichlet process
            'sigma': sigma,
            'tau': tau,
        }

        self.w_0_mh_sampler = MetropolisHastings(initial_step_size=0.1)
        self.w_k_mh_sampler = MetropolisHastings(initial_step_size=0.1)
        self.w_0_proportion_sampler = HamiltonMonteCarlo(is_independent=False, initial_step_size=0.1)

        if cmr == 'gamma':
            log_v = lambda s: - torch.log(s) - s
            log_u = lambda x: - (torch.lgamma(torch.tensor([alpha]))) + (alpha - 1.0) * torch.log(x) - x
            dlog_v = lambda s: - 1.0 / s - 1.0
            dlog_u = lambda x: (alpha - 1.0) * 1. / x - 1.
            self.log_v = log_v
            self.log_u = log_u
            self.dlog_v = dlog_v
            self.dlog_u = dlog_u

    def one_step(self):
        self.state['m'] = compute_m(self.state['z'], self.state['c'], self.max_K)
        self.state['n'] = compute_n(self.state['m'])
        self.update_w_0_total()

        self.update_w_0_proportion()
        self.update_w_total()
        self.update_w_proportion()

        self.update_pi()
        self.update_z()
        self.update_c()

    def fit(self, epochs):
        for i in range(epochs):
            print('number of epoch', i)
            self.one_step()
            #print(self.state['c']._values()[0:20])
            #print(self.state['pi'][0:10])

    def update_w_proportion(self):
        log_w_bar = self.state['log_w_total'][1:self.active_K]
        log_w = sample_w_proportion(self.state['m'][1:self.active_K], self.state['log_w_0'], log_w_bar)
        self.state['log_w'] = torch.cat([torch.unsqueeze(self.state['log_w_0'], dim=0) - torch.log(torch.sum(torch.exp(self.state['log_w_0']))),
                                         log_w, self.state['log_w'][self.active_K::]], dim=0)

    def update_pi(self):
        # the number of links in each clusters, first row corresponds to cluster 0.
        pi = sample_pi(self.state['n'][1:self.active_K], self.hyper_paras['gamma'])
        self.state['pi'] = torch.cat([pi, torch.zeros(self.max_K - self.active_K)], dim=0)
        #print('pi', self.state['pi'])

    def update_c(self):
        c = compute_c(self.state['pi'][0:self.active_K], self.state['log_w'][0:self.active_K], self.state['z'])
        self.state['c'],  self.state['log_w'], self.active_K = add_k(c, self.state['log_w'], self.active_K, self.max_K, self.hyper_paras['alpha'])

    def update_z(self):
        self.state['z'] = compute_z(self.state['log_w'][0:self.active_K], self.state['c'], self.graph_sparse)

    def update_w_0_total(self):
        log_prob_fn = functools.partial(log_prob_wrt_w_0_total, log_w_k=self.state['log_w'][1:self.active_K],
                                        log_u=self.log_u)
        new_log_w_0_total = self.w_0_mh_sampler.one_step(state=self.state['log_w_0_total'], log_prob_fn=log_prob_fn)
        self.state['log_w_0_total'] = new_log_w_0_total

    def update_w_total(self):
        log_prob_fn = functools.partial(log_prob_wrt_w_k_total, n=self.state['n'][1:self.active_K],
                                        log_w_0_total=self.state['log_w_0_total'],
                                        pi=self.state['pi'][1:self.active_K])
        new_log_w_total = self.w_k_mh_sampler.one_step(state=self.state['log_w_total'][1:self.active_K],
                                                       log_prob_fn=log_prob_fn)
        self.state['log_w_total'] = torch.cat([self.state['log_w_total'][0:1],
                                               new_log_w_total,
                                               self.state['log_w_total'][self.active_K::]],
                                              dim=0)

    def update_w_0_proportion(self):
        w_0_proportion = self.state['log_w_0'][1::]

        log_prob_fn = functools.partial(log_prob_wrt_w_0_proportional,
                                        m=self.state['m'][1:self.active_K],
                                        w_0_total=torch.exp(self.state['log_w_0_total']),
                                        log_u=self.log_u,
                                        log_v=self.log_v)
        d_log_prob_fn = functools.partial(d_log_prob_wrt_w_0_proportional,
                                          m=self.state['m'][1:self.active_K],
                                          w_0_total=torch.exp(self.state['log_w_0_total']),
                                          dlog_u=self.dlog_u,
                                          dlog_v=self.dlog_v, )
        tmp = self.w_0_proportion_sampler.one_step(w_0_proportion,
                                                   log_prob_fn=log_prob_fn,
                                                   d_log_prob_fn=d_log_prob_fn)
        new_log_w = torch.exp(tmp) / torch.sum(torch.exp(tmp))

        self.state['log_w_0'] = torch.cat([self.state['log_w_0'][0:1],
                                           new_log_w],
                                          dim=0)

    def sample(self):
        total_weight = self.state['pi'][1:self.active_K] * self.state['log_w_total'][1:self.active_K]
        print('total_weight', total_weight,)
        total_number_edges = torch.maximum(torch.poisson(total_weight), torch.ones_like(total_weight))
        print('total_number_edges', total_number_edges)

        edge_index = torch.cat([Categorical(logits=self.state['log_w'][i]).sample([2, int(total_number_edges[i-1].item())]) for i in range(1, self.active_K)], dim=1)

        return edge_index