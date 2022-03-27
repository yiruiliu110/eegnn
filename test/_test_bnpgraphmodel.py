"""
This script tests the init function of BNPGraphModel.
"""
import torch

from estimation.graph_model import BNPGraphModel

graph = [[0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]]


graph = torch.tensor(graph).to_sparse(2)
print(graph)

alpha = 300.
tau = 1.0
gamma = 5.0
sigma = 0.5


#### test 0
model = BNPGraphModel(graph, alpha, tau, gamma, sigma, initial_K=3, max_K=100)
print(model.state)

#### test 1
model.update_w_proportion()
print('log_w', model.state['log_w'])

#### test 2
model.update_pi()
print('pi', model.state['pi'])

#### test 3
model.update_c()
print('c', model.state['c'])

#### test 4
model.update_z()
print('z', model.state['z'])

#### test 5
print('log_w_0', model.state['log_w_0'])
print('log_w_0_total', model.state['log_w_0_total'])
for i in range(1000):
    model.update_w_0_total()
print('log_w_0', model.state['log_w_0'])
print('log_w_0_total', model.state['log_w_0_total'])

#### test 6
print('log_w_total', model.state['log_w_total'])
for i in range(1000):
    model.update_w_total()
print('log_w_total', model.state['log_w_total'])
print('log_w_0_total', model.state['log_w_0_total'])


#### test 7
print('log_w_total', model.state['log_w_total'])
for i in range(10):
    model.update_w_0_proportion()
    print('log_w_total', model.state['log_w_total'])

