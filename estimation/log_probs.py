import torch


def log_prob_wrt_w_0_total(log_w_0_total, log_w_k_total, log_u):
    w_0_total = torch.exp(log_w_0_total)
    return log_u(w_0_total) + torch.sum(w_0_total * log_w_k_total - torch.lgamma(w_0_total))


def log_prob_wrt_w_k_total(log_w_k_total, n, log_w_0_total, pi):
    # log_w_k_total (actiave_K -1,)
    w_k_total = torch.exp(log_w_k_total)
    w_0_total = torch.exp(log_w_0_total)
    return (2.0 * n + w_0_total - 1.0) * log_w_k_total - w_k_total - w_k_total * w_k_total * pi


def log_prob_wrt_w_0_proportional(w_0_proportion_nonstar, m, w_0_total, log_u, log_v):
    # w_0_proportion (|V|, )
    w_0_nonstar = w_0_proportion_nonstar[0:-1] * w_0_total
    w_0_star = w_0_proportion_nonstar[-1] * w_0_total
    return torch.sum(torch.lgamma(m + w_0_nonstar) - torch.lgamma(w_0_nonstar), dim=0) \
           + log_v(w_0_nonstar) + log_u(w_0_star)  # outputshape: (|V|, )


def d_log_prob_wrt_w_0_proportional(w_0_proportion_nonstar, m, w_0_total, dlog_u, dlog_v):
    # w_0_proportion (|V|, )
    w_0_nonstar = w_0_proportion_nonstar[0:-1] * w_0_total
    w_0_star = w_0_proportion_nonstar[-1] * w_0_total
    return torch.sum(torch.digamma(m + w_0_nonstar) - torch.digamma(w_0_nonstar), dim=0) \
           + dlog_v(w_0_nonstar) - dlog_u(w_0_star)  # outputshape: (|V|, )


def forward_to_log_softmax_weight(log_w):
    return log_w[0:-1] - log_w[-1]


def back_to_log_weight(log_w_prime):
    unnormalzied_log_w = torch.cat([log_w_prime, torch.tensor([0.0])], dim=0)
    log_w = unnormalzied_log_w - torch.log(torch.sum(torch.exp(unnormalzied_log_w + 1e-14)))
    return log_w


def back_to_weight(log_w_prime):
    return torch.exp(back_to_log_weight(log_w_prime))



