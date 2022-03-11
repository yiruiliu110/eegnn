import torch


def is_accepted(log_accept_ratio):
# If proposed state reduces likelihood: randomly accept.
    # If proposed state increases likelihood: always accept.
    # I.e., u < min(1, accept_ratio),  where u ~ Uniform[0,1)
    #       ==> log(u) < log_accept_ratio
    log_uniform = torch.log(
        samplers.uniform(
            shape=prefer_static.shape(proposed_results.target_log_prob),
            )
    result = log_uniform < log_accept_ratio
    return result