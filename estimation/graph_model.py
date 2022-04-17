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

        self.w_0_mh_sampler = MetropolisHastings(initial_step_size=0.1)
        self.w_k_mh_sampler = MetropolisHastings(initial_step_size=1.0)
        self.w_0_proportion_sampler = HamiltonMonteCarlo(is_independent=True, initial_step_size=0.01, num_leapfrog_steps=2)

        if cmr == 'gamma':
            log_v = lambda s: - torch.log(s) - s
            log_u = lambda x: - (torch.lgamma(torch.tensor([alpha]))) + (alpha - 1.0) * torch.log(x) - x
            dlog_v = lambda s: - 1.0 / s - 1.0
            dlog_u = lambda x: (alpha - 1.0) * 1. / x - 1.
            self.log_v = log_v
            self.log_u = log_u
            self.dlog_v = dlog_v
            self.dlog_u = dlog_u

    def one_step(self, update_number_cluster=True):
        self.state['m'] = compute_m(self.state['z'], self.state['c'], self.max_K)
        self.state['n'] = compute_n(self.state['m'])

        self.update_w_0_total()
        self.update_w_0_proportion()

        self.update_w_total()
        self.update_w_proportion()

        self.update_pi()

        self.update_z()
        self.update_c()

        if update_number_cluster:
            self.adjust_cluster_number()

    def fit(self, epochs):
        for i in range(epochs):
            print('number of epoch', i)
            self.one_step()
            print('log likelihood', self.log_likelihood())

    def update_w_proportion(self):
        log_w = sample_w_proportion(self.state['m'][1:self.active_K], self.state['log_w_0'],
                                    self.state['log_w_total'][1:self.active_K])
        self.state['log_w'] = torch.cat([torch.unsqueeze(self.state['log_w_0'], 0), log_w, self.state['log_w'][self.active_K::]], dim=0)

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
        self.state['c'], self.active_K, remaining_indices = add_k(
            self.state['c'],
            self.active_K,
            self.max_K,
            self.hyper_paras['gamma']
        )
        for item in ['log_w_total', 'log_w', 'pi', 'm', 'n']:
            self.state[item] = switch(self.state[item], remaining_indices, self.max_K)
        print('active_K', self.active_K)

    def update_w_0_total(self):
        log_prob_fn = functools.partial(log_prob_wrt_w_0_total, log_w_k_total=self.state['log_w_total'][1:self.active_K],
                                        log_u=self.log_u)
        self.state['log_w_0_total'] = self.w_0_mh_sampler.one_step(state=self.state['log_w_0_total'],
                                                                   log_prob_fn=log_prob_fn)

    def update_w_total(self):
        # cluster 0
        new_log_w_total_0 = torch.log(Gamma(concentration=torch.exp(self.state['log_w_0_total']) + 1e-10, rate=1.).sample())
        # other clusters
        log_prob_fn = functools.partial(log_prob_wrt_w_k_total, n=self.state['n'][1:self.active_K]/2,
                                        log_w_0_total=self.state['log_w_0_total'],
                                        pi=self.state['pi'][1:self.active_K])
        new_log_w_total = self.w_k_mh_sampler.one_step(state=self.state['log_w_total'][1:self.active_K],
                                                       log_prob_fn=log_prob_fn)
        self.state['log_w_total'] = torch.cat([new_log_w_total_0,
                                               new_log_w_total,
                                               torch.log(torch.squeeze(
                                                   Gamma(concentration=torch.exp(self.state['log_w_0_total']) + 1e-10, rate=1.).sample(
                                                   (self.max_K - self.active_K,))))],   #self.max_K needs to be larger than self.active_K
                                              dim=0)

    def update_w_0_proportion(self):
        log_prob_fn = functools.partial(log_prob_wrt_w_0_proportional,
                                        m=self.state['m'][1:self.active_K, 0:-1],
                                        w_0_total=torch.exp(self.state['log_w_0_total']),
                                        log_u=self.log_u, log_v=self.log_v)
        d_log_prob_fn = functools.partial(d_log_prob_wrt_w_0_proportional,
                                          m=self.state['m'][1:self.active_K, 0:-1],
                                          w_0_total=torch.exp(self.state['log_w_0_total']),
                                          dlog_u=self.dlog_u, dlog_v=self.dlog_v)

        tmp = self.w_0_proportion_sampler.one_step(
            forward_to_log_softmax_weight(self.state['log_w_0']),
            log_prob_fn=lambda inputs: log_prob_fn(back_to_weight(inputs)),
            d_log_prob_fn=lambda inputs: self.gradient(d_log_prob_fn, back_to_weight(inputs))
        )

        self.state['log_w_0'] = back_to_log_weight(tmp)

    @staticmethod
    def gradient(d_log_prob_fn, weight):
        derivative = d_log_prob_fn(weight)
        grad = derivative * weight[0:-1]
        return grad - torch.sum(grad * weight[0:-1])

    def sample(self):
        total_weight = self.state['pi'][1:self.active_K] * torch.exp(self.state['log_w_total'][1:self.active_K]) ** 2
        total_number_edges = torch.poisson(total_weight)

        self.total_number_sampled_edges = int(torch.sum(total_number_edges))
        print('total_number_edges', self.total_number_sampled_edges, total_number_edges)

        edge_index = torch.cat(
            [Categorical(logits=self.state['log_w'][i][1::]).sample([2, int(total_number_edges[i-1].item())]) for i in
             range(1, self.active_K) if int(total_number_edges[i-1].item()) > 0], dim=1)
        return self.to_symmetric(edge_index)

    @staticmethod
    def to_symmetric(edge_index):
        """
        indices = edge_index._indices()
        values = edge_index._values()

        indices_0, indices_1 = indices[0], indices[1]

        new_index = torch.cat([torch.unsqueeze(torch.cat([indices_0, indices_1], dim=0), dim=0),
                               torch.unsqueeze(torch.cat([indices_1, indices_0], dim=0), dim=0)],
                              dim=0)
        new_values = torch.cat([values, values], dim=0)

        return torch.sparse_coo_tensor(new_index, new_values, edge_index.size())
        """
        indices_0, indices_1 = edge_index[0], edge_index[1]
        new_index = torch.cat([torch.unsqueeze(torch.cat([indices_0, indices_1], dim=0), dim=0),
                               torch.unsqueeze(torch.cat([indices_1, indices_0], dim=0), dim=0)],
                              dim=0)
        return new_index

    def log_likelihood(self):
        log_prob = torch.sum(torch.log(self.state['pi'][1:self.active_K] + 1e-10) * self.state['n'][1:self.active_K])

        log_prob += torch.sum((self.state['log_w'][1:self.active_K] - torch.unsqueeze(self.state['log_w_total'][1:self.active_K], 1)) * self.state['m'][1:self.active_K])

        return log_prob

    def sample_conditonal(self):
        logits = self.state['log_w'][0:self.active_K] + torch.unsqueeze(torch.log(self.state['pi'][0:self.active_K] + 1e-15), 1)
        c = Categorical(logits=torch.transpose(logits[:, 0:-1], 0, 1)).sample()

        dist_pras = torch.index_select(self.state['log_w'][0:self.active_K], 0, c)
        nodes = Categorical(logits=dist_pras).sample()

        outputs = torch.cat([torch.unsqueeze(torch.arange(0, self.node_number), 0), torch.unsqueeze(nodes, 0)], 0)
        return outputs

    def compute_mean(self, number_of_samples: int = 1000):
        results_pi = []
        results_log_w = []
        for i in range(number_of_samples):
            print('number of samples', i)
            self.one_step(update_number_cluster=False)
            print('log likelihood', self.log_likelihood())
            results_pi.append(torch.unsqueeze(self.state['pi'][0:self.active_K], 0))
            results_log_w.append(torch.unsqueeze(self.state['log_w'][0:self.active_K], 0))

        mean_pi = torch.mean(torch.cat(results_pi, dim=0), dim=0)
        mean_log_w = torch.mean(torch.cat(results_log_w, dim=0), dim=0)

        return mean_pi, mean_log_w





