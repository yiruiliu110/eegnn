"""
test mh using a normal dist and multivariate normal dist, and two independ normal
"""
import torch
import seaborn as sns
from estimation.mh import MetropolisHastings
import matplotlib.pyplot as plt
log_prob_fn = torch.distributions.normal.Normal(0., 1.).log_prob

sampling = torch.tensor([1.0])

MH = MetropolisHastings()

result = []
for i in range(10000):
    sampling = MH.one_step(sampling, log_prob_fn)
    result += [sampling]

sns.displot(torch.tensor(result[-500:-1]))
plt.show()


#########
# test 2
########
log_prob_fn = torch.distributions.normal.Normal(torch.tensor([0., 1.0]), 1.).log_prob

sampling = torch.cat([1.0, 0.0])

MH = MetropolisHastings()

result = []
for i in range(10000):
    sampling = MH.one_step(sampling, log_prob_fn)
    result += [sampling]

result = torch.cat([torch.tensor(item) for item in result[-500:-1]], dim=0)

sns.displot(result[:, 1])
plt.show()