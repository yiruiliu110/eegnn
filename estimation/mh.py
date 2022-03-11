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

        self.last_step_log_prob = None

        self.number_steps = 0

    def one_step(self, state, log_prob_fn):
        with torch.no_grad():
            if self.last_step_log_prob is None:
                self.last_step_log_prob = log_prob_fn(state)
            if self.step_size is None:
                self.step_size = torch.ones(state.size()) * self.initial_step_size
            proposed_state = state + torch.normal(0.0, 1.0, state.size()) * self.step_size

            propose_log_prob = log_prob_fn(proposed_state)

            log_accept_ratio = propose_log_prob - self.last_step_log_prob

            accepted = is_accepted(log_accept_ratio)

            accepted, _ = torch.broadcast_tensors(accepted, state)

            new_state = torch.where(accepted, proposed_state, state)

            self.last_step_log_prob = torch.where(accepted, propose_log_prob, self.last_step_log_prob)

            self.number_steps += 1

            return new_state







