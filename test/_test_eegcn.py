from torch_geometric.datasets import Planetoid

from estimation.eegcn import EEGCNConv
from estimation.graph_model import BNPGraphModel

dataset = Planetoid(root='/Users/yiruiliu/PycharmProjects/pythonProject/Cora', name='Cora')
print(dataset)

import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.number_layer = 24
        self.conv1 = EEGCNConv(dataset.num_node_features, 16)
        self.conv_mid = {}
        for i in range(62):
            self.conv_mid[i] = EEGCNConv(16, 16)
        self.conv2 = EEGCNConv(16, dataset.num_classes)

        self.egg1 = EEGCNConv(dataset.num_node_features, 16)
        self.egg_mid = {}
        for i in range(62):
            self.egg_mid[i] = EEGCNConv(16, 16)
        self.egg2 = EEGCNConv(16, dataset.num_classes)

        self.graph = None
        self.estimated_graph = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.graph is None:
            number_of_edges = int(edge_index.size()[1])
            number_of_nodes = int(torch.max(edge_index).item()) + 1
            self.graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                            [number_of_nodes, number_of_nodes])

        if self.estimated_graph is None:
            self.estimated_graph = BNPGraphModel(self.graph, alpha=1.0, tau=1.0, gamma=1.0, sigma=0.5, initial_K=50, max_K=200)
            self.estimated_graph.fit(1000)

        with torch.no_grad():
            edge_index_sample = self.estimated_graph.sample()[:, 0:500]
        x_0 = self.egg1(x, edge_index_sample)
        x_0 = F.relu(x_0)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = 0.8 * x + 0.2 * x_0

        for i in range(62):
            with torch.no_grad():
                edge_index_sample = self.estimated_graph.sample()[:, torch.randperm(self.estimated_graph.total_number_sampled_edges)[:1000]]
            x_0 = self.egg_mid[i](x_0, edge_index_sample)
            x_0 = F.relu(x_0)
            x = self.conv_mid[i](x, edge_index)
            x = 0.8 * x + 0.2 * x_0
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)

        x_0 = self.egg2(x_0, edge_index_sample)
        x_0 = F.relu(x_0)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = 0.8 * x + 0.2 * x_0

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')