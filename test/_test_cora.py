import torch

from estimation.graph_model import BNPGraphModel

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/Users/yiruiliu/PycharmProjects/pythonProject/Cora', name='Cora')
print(len(dataset))  # 1
print(dataset.num_classes)  # 7
print(dataset.num_features)  # 1433

# Here, the dataset contains only a single, undirected citation graph:
data = dataset[0]  # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708],
#                           val_mask=[2708], test_mask=[2708])
print(data)


number_of_nodes = data.x.size()[0]
number_of_edges = data.edge_index.size()[1]
graph = torch.sparse_coo_tensor(data.edge_index, torch.ones(number_of_edges), [number_of_nodes, number_of_nodes])
print(graph)

alpha = 10.0
tau = 1.0
gamma = 1.0
sigma = 0.5

#### test 0
model = BNPGraphModel(graph, alpha, tau, gamma, sigma, initial_K=50, max_K=200)
#### test 8
model.fit(1000)

model.sample()
