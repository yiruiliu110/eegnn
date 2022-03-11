"""
HMC
"""
from estimation.mh import MetropolisHastings
import torch


class HamiltonMonteCarlo(MetropolisHastings):

    def __init__(self, initial_step_size=0.1, num_leapfrog_steps=10, is_independent=True):
        super(HamiltonMonteCarlo, self).__init__(initial_step_size)
        self.num_leapfrog_steps = num_leapfrog_steps
        self.is_independent = is_independent  # is True, elements of states are independent

    def one_step(self, state, log_prob_fn, d_log_prob_fn):
        with torch.no_grad():
            self.build_initials(state, log_prob_fn)
            init_p = torch.normal(0.0, 1.0, state.size())
            p = init_p

            proposed_state = state

            i = 0
            while i < self.num_leapfrog_steps:
                p = p + 0.5 * self.step_size * d_log_prob_fn(proposed_state)
                proposed_state = proposed_state + self.step_size * p
                p = p + 0.5 * self.step_size * d_log_prob_fn(proposed_state)
                i += 1

            propose_energy = log_prob_fn(proposed_state) - 0.5 * p * p
            init_energy = log_prob_fn(state) - 0.5 * init_p * init_p

            log_accept_ratio = propose_energy - init_energy

            if not self.is_independent:
                log_accept_ratio = torch.sum(log_accept_ratio)

            new_state, _ = self.mh_accept(proposed_state, state, log_accept_ratio)

            return new_state






