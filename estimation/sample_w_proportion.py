import torch
from torch.distributions import Gamma


def sample_w_proportion(m, log_w_0, log_w_bar):
    """
    sample w_proportion
    :param m: the matrix containing n_{k,i}.  (active_K - 1) X (number_of_nodes).  cluster 0 is NOT included.
    :param log_w_0: (number_of_nodes + 1, )
    :param w_bar: (active_K - 1, )
    :return: log_w : (active_K - 1) X (number_of_nodes + 1)
    """
    torch.cat([torch.zeros(m.size()[0], 1), m], dim=1)
    concentration = torch.cat([torch.zeros(m.size()[0], 1), m], dim=1) + torch.exp(log_w_0)
    w_tmp = Gamma(concentration=concentration, rate=1.).sample()  # max_K X number_nodes
    log_w = torch.log(w_tmp + 1e-15) - torch.log(torch.sum(w_tmp, dim=1, keepdim=True) + 1e-15) + torch.unsqueeze(log_w_bar, dim=1)
    return log_w


if __name__ == "__main__":
    active_k = 10
    num_nodes = 15
    results = sample_w_proportion(m=torch.randint(0, 5, (active_k - 1, num_nodes)),
                                  log_w_0=torch.ones(num_nodes + 1),
                                  w_bar=10.0 * torch.ones(active_k - 1))
    print(results)
