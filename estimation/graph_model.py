"""
This file contain the class for Bayesian nonparametric graph model
"""
import functools

import torch
from torch.distributions import Categorical, Gamma

from estimation.add_k import add_k, switch
from estimation.build_initials import build_initials, compute_n
from estimation.compute_m import compute_m
from estimation.hmc import HamiltonMonteCarlo
from estimation.log_probs import log_prob_wrt_w_0_total, log_prob_wrt_w_k_total, log_prob_wrt_w_0_proportional, \
    d_log_prob_wrt_w_0_proportional, forward_to_log_softmax_weight, back_to_log_weight, back_to_weight
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
        graph_dense = graph.to_dense()
        print('is symmetric: (0 is yes)', torch.sum(graph_dense - torch.transpose(graph_dense, 0, 1)))
        print('is self connected: 0 is no', torch.sum(torch.diag(graph_dense, 0)))
        # graph_dense.fill_diagonal_(1)

        dense_graph = torch.triu(graph_dense)
        self.graph_sparse = dense_graph.to_sparse(2)

        self.node_number = dense_graph.shape[0]
        self.edge_number = torch.sum(dense_graph).to(int)
        print(f"Num Nodes : {self.node_number} \t Num Edges : {self.edge_number}")

        # number of clusters
        self.max_K = max_K
        self.active_K = initial_K

        # Step 0: Initialization
        self.state = build_initials(initial_K, max_K, self.graph_sparse, self.node_number, self.edge_number)

        # hyper parameters
        self.hyper_paras = {
            'alpha': alpha,  # hyper parameter for nodes
            'gamma': gamma,  # the hyper_para for cluster proportion
            'sigma': sigma,  # 0.5
            'tau': tau,  # 1.0
        }

        self.w_0_mh_sampler = MetropolisHastings(initial_step_size=1.0)
        self.w_k_mh_sampler = MetropolisHastings(initial_step_size=1.0)
        self.w_0_proportion_sampler = HamiltonMonteCarlo(is_independent=False, initial_step_size=1.0)

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

        self.adjust_cluster_number()

    def fit(self, epochs):
        for i in range(epochs):
            print('number of epoch', i)
            self.one_step()

    def update_w_proportion(self):
        log_w = sample_w_proportion(self.state['m'][0:self.active_K], self.state['log_w_0'],
                                    self.state['log_w_total'][0:self.active_K])
        self.state['log_w'] = torch.cat([log_w, self.state['log_w'][self.active_K::]], dim=0)

    def update_pi(self):
        # the number of links in each clusters, first row corresponds to cluster 0.
        pi = sample_pi(self.state['n'][0:self.active_K], self.hyper_paras['gamma'])
        self.state['pi'] = torch.cat([pi, torch.zeros(self.max_K - self.active_K)], dim=0)

    def update_c(self):
        self.state['c'] = compute_c(self.state['pi'][0:self.active_K], self.state['log_w'][0:self.active_K],
                                    self.state['z'])

    def update_z(self):
        self.state['z'] = compute_z(self.state['log_w'][0:self.active_K], self.state['pi'], self.state['c'])

    def adjust_cluster_number(self):
        self.state['c'], self.active_K, remaining_indices, deleting_indices = add_k(
            self.state['c'],
            self.active_K,
            self.max_K,
            self.hyper_paras['gamma']
        )
        for item in ['log_w_total', 'log_w', 'pi', 'm', 'n']:
            self.state[item] = switch(self.state[item], remaining_indices, deleting_indices)

    def update_w_0_total(self):
        log_prob_fn = functools.partial(log_prob_wrt_w_0_total, log_w_k_total=self.state['log_w_total'][1:self.active_K],
                                        log_u=self.log_u)
        self.state['log_w_0_total'] = self.w_0_mh_sampler.one_step(state=self.state['log_w_0_total'],
                                                                   log_prob_fn=log_prob_fn)

    def update_w_total(self):
        # cluster 0
        new_log_w_total_0 = Gamma(concentration=self.state['log_w_0_total'], rate=1.).sample()
        # other clusters
        log_prob_fn = functools.partial(log_prob_wrt_w_k_total, n=self.state['n'][1:self.active_K],
                                        log_w_0_total=self.state['log_w_0_total'],
                                        pi=self.state['pi'][1:self.active_K])
        new_log_w_total = self.w_k_mh_sampler.one_step(state=self.state['log_w_total'][1:self.active_K],
                                                       log_prob_fn=log_prob_fn)
        self.state['log_w_total'] = torch.cat([new_log_w_total_0,
                                               new_log_w_total,
                                               torch.squeeze(
                                                   Gamma(concentration=self.state['log_w_0_total'], rate=1.).sample(
                                                   (self.max_K - self.active_K,)))],
                                              dim=0)

    def update_w_0_proportion(self):
        log_prob_fn = functools.partial(log_prob_wrt_w_0_proportional,
                                        m=self.state['m'][1:self.active_K],
                                        w_0_total=torch.exp(self.state['log_w_0_total']),
                                        log_u=self.log_u, log_v=self.log_v)
        d_log_prob_fn = functools.partial(d_log_prob_wrt_w_0_proportional,
                                          m=self.state['m'][1:self.active_K],
                                          w_0_total=torch.exp(self.state['log_w_0_total']),
                                          dlog_u=self.dlog_u, dlog_v=self.dlog_v)

        tmp = self.w_0_proportion_sampler.one_step(
            forward_to_log_softmax_weight(self.state['log_w_0']),
            log_prob_fn=lambda inputs: log_prob_fn(back_to_weight(inputs)),
            d_log_prob_fn=lambda inputs: self.gradient(d_log_prob_fn, back_to_weight(inputs))
        )

        self.state['log_w_0'] = back_to_log_weight(tmp)

    @staticmethod
    def gradient(d_log_prob_fn, inputs):
        weight = back_to_weight(inputs)
        derivative = d_log_prob_fn(weight)
        return (derivative[0:-1] - torch.sum(derivative * weight)) * weight[0:-1]

    def sample(self):
        total_weight = self.state['pi'][1:self.active_K] * torch.exp(self.state['log_w_total'][1:self.active_K]) ** 2
        total_number_edges = torch.maximum(torch.poisson(total_weight), torch.ones_like(total_weight))
        self.total_number_sampled_edges = int(torch.sum(total_number_edges))
        print('total_number_edges', self.total_number_sampled_edges)

        edge_index = torch.cat(
            [Categorical(logits=self.state['log_w'][i][1::]).sample([2, int(total_number_edges[i - 1].item())]) for i in
             range(1, self.active_K)], dim=1)
        return edge_index
