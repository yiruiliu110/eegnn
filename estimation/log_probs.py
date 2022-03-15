import torch


def log_prob_wrt_w_0_total(log_w_0, log_w_k, log_u):
    w_0 = torch.exp(log_w_0)
    return log_u(w_0) + torch.sum(w_0 * log_w_k - torch.lgamma(w_0))


def log_prob_wrt_w_k_total(log_w_k_total, n, log_w_0_total, pi):
    # log_w_k_total (actiave_K -1,)
    w_k_total = torch.exp(log_w_k_total)
    w_0_total = torch.exp(log_w_0_total)
    return (2.0 * n + w_0_total - 1.0) * log_w_k_total - w_k_total - w_k_total * w_k_total * pi


def log_prob_wrt_w_0_proportional(w_0_proportion, m, w_0_total, log_u, log_v):
    # w_0_proportion (|V|, )
    tmp = torch.exp(w_0_proportion)
    sum_tmp = torch.sum(tmp)
    w_0_nonstar = tmp * (w_0_total / (sum_tmp + 1.))
    w_0_star = w_0_total / (sum_tmp + 1.)

    return torch.sum(torch.lgamma(m + w_0_nonstar) - torch.lgamma(w_0_nonstar), dim=0) \
           + log_v(w_0_nonstar) + log_u(w_0_star) # outputshape: (|V|, )


def d_log_prob_wrt_w_0_proportional(w_0_proportion, m, w_0_total, dlog_u, dlog_v):
    # w_0_proportion (|V|, )
    tmp = torch.exp(w_0_proportion)
    sum_tmp = torch.sum(tmp)
    w_0_nonstar = tmp * (w_0_total / (sum_tmp + 1.))
    w_0_star = w_0_total / (sum_tmp + 1.)

    tmp = torch.sum(torch.digamma(m + w_0_nonstar) - torch.digamma(w_0_nonstar), dim=0) \
          + dlog_v(w_0_nonstar)  - dlog_u(w_0_star)  # outputshape: (|V|, )

    result = tmp * w_0_proportion * (1.0 - w_0_proportion)

    return result

