from torch_geometric.datasets import Planetoid
from estimation.model import EEGCNModel
import torch
import torch.nn.functional as F


dataset = Planetoid(root='/Users/yiruiliu/PycharmProjects/pythonProject/Cora', name='Cora')
print(dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0]#.to(device)
data = data.to(device)
model = EEGCNModel(data, number_layer=24).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    out = model(data)#data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')