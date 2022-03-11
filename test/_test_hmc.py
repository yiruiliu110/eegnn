"""
test hmc using  two independent normal
"""
import torch
import seaborn as sns

from estimation.hmc import HamiltonMonteCarlo
import matplotlib.pyplot as plt


log_prob_fn = torch.distributions.normal.Normal(torch.tensor([0., 1.0]), 1.).log_prob
d_log_prob_fn = lambda x: - x

sampling = torch.tensor([1.0, 0.0])

HMC = HamiltonMonteCarlo(is_independent=False)

result = []
for i in range(20000):
    sampling = HMC.one_step(sampling, log_prob_fn, d_log_prob_fn )
    result += [sampling]

result = torch.cat([torch.unsqueeze(item, 1) for item in result[-5000:-1]], dim=1)

sns.displot(result[0])
plt.show()
sns.displot(result[1])
plt.show()