

import torch
from torch.distributions import constraints, Poisson, Categorical


class TruncatedPoisson(Poisson):
    """
        Truncated Poisson random variables sampling
    """
    arg_constraints = {'rate': constraints.nonnegative}
    support = constraints.nonnegative_integer

    def __init__(self, rate, validate_args=None):
        super(TruncatedPoisson, self).__init__(rate, validate_args)

    def sample(self, limit=10):

        with torch.no_grad():
            weights = self.log_prob(torch.unsqueeze(torch.range(1, limit), dim=1))  # limit X shape_of_rates
            categorial_distribution = Categorical(logits=torch.transpose(weights, dim0=0, dim1=1))
            sample = categorial_distribution.sample()
            return sample + 1

