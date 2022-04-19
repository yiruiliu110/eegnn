import torch



from torch_geometric.datasets import Planetoid

from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel

dataset = Planetoid(root='/Users/yiruiliu/PycharmProjects/pythonProject/data/Cora', name='Cora')
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

alpha = 1.0
tau = 1.0

gamma = 1.0
sigma = 0.5

#### test 0
model = BNPGraphModel(graph, alpha, tau, gamma, sigma, initial_K=100, max_K=150)
print(model.state)
#### test 8
model.fit(1000)

model.sample()

model.sample_conditonal()

mean_pi, mean_log_w = model.compute_mean(1000)
print(mean_pi, mean_log_w)

index, weight = compute_dege_index_and_weight(mean_pi, mean_log_w[:, 0:-1])