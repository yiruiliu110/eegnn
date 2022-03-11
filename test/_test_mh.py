"""
test mh using a normal dist and multivariate normal dist, and two independ normal
"""
import torch
import seaborn as sns
from estimation.mh import MetropolisHastings
import matplotlib.pyplot as plt
log_prob_fn = torch.distributions.normal.Normal(0., 1.).log_prob

sampling = torch.tensor([1.0])

MH = MetropolisHastings(0.05)

result = []
for i in range(20000):
    sampling = MH.one_step(sampling, log_prob_fn)
    result += [sampling]

sns.displot(torch.tensor(result[-5000:-1]))
plt.show()


#########
# test 2
########
log_prob_fn = torch.distributions.normal.Normal(torch.tensor([0., 1.0]), 1.).log_prob

sampling = torch.tensor([1.0, 0.0])

MH = MetropolisHastings()

result = []
for i in range(20000):
    sampling = MH.one_step(sampling, log_prob_fn)
    result += [sampling]

result = torch.cat([torch.unsqueeze(item, 1) for item in result[-5000:-1]], dim=1)

sns.displot(result[0])
plt.show()
sns.displot(result[1])
plt.show()