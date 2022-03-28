import torch


def is_accepted(log_accept_ratio):
    with torch.no_grad():
        # If proposed state reduces likelihood: randomly accept.
        # If proposed state increases likelihood: always accept.
        # I.e., u < min(1, accept_ratio),  where u ~ Uniform[0,1)
        #       ==> log(u) < log_accept_ratio
        log_uniform = torch.log(
            torch.rand(
                size=log_accept_ratio.size()),
                )
        result = log_uniform < log_accept_ratio
        return result


class MetropolisHastings:

    def __init__(self, initial_step_size=0.1):
        self.initial_step_size = initial_step_size
        self.step_size = None

        self.state_shape = None

        self.number_steps = 0

    def one_step(self, state, log_prob_fn):
        with torch.no_grad():
            self.build_initials(state, log_prob_fn)
            proposed_state = state + torch.normal(0.0, 1.0, state.size()) * self.step_size

            propose_log_prob = log_prob_fn(proposed_state)
            last_step_log_prob = log_prob_fn(state)

            log_accept_ratio = propose_log_prob - last_step_log_prob

            new_state, accepted = self.mh_accept(proposed_state, state, log_accept_ratio)

            self.last_step_log_prob = torch.where(accepted, propose_log_prob, last_step_log_prob)

            self.number_steps += 1

            return new_state

    def build_initials(self, state, log_prob_fn):
        with torch.no_grad():
            if self.step_size is None:
                self.step_size = self.initial_step_size

    def mh_accept(self, proposed_state, state, log_accept_ratio):
        with torch.no_grad():

            accepted = is_accepted(log_accept_ratio)

            accepted, _ = torch.broadcast_tensors(accepted, state)

            new_state = torch.where(accepted, proposed_state, state)

            return new_state, accepted








