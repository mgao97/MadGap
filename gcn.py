import os
import copy
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig('figs/cora_gcn.png')


def accuracy(y_pred, y_true):
    
    correct = y_pred.eq(y_true).double()
    correct = correct.sum().item()

    accuracy_value = correct / len(y_true)
    # 计算准确率的标准差
    squared_error = (correct - accuracy_value * len(y_true))**2
    variance = squared_error / len(y_true)
    accuracy_std = np.sqrt(variance)

    return accuracy_value, accuracy_std

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    print()
    print('='*80)

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


    model = GCN(dataset.num_features, 512, dataset.num_classes)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    num_epochs = 200
    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

    
    
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)
            val_loss = cross_entropy(out[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss)


            val_pred = out.argmax(dim=1)
            val_acc,_ = accuracy(val_pred[data.val_mask], data.y[data.val_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figs/loss_curve_gcn.png')

    model.eval()
    model = best_model
    out = model(data.x, data.edge_index)
    test_pred = out.argmax(dim=1)
    test_acc, test_acc_std = accuracy(test_pred[data.test_mask], data.y[data.test_mask])
    print(f'Test Accuracy: {test_acc:.4f}, Test Acc std: {test_acc_std:.4f}')

    # model.eval()
    best_out = model(data.x, data.edge_index)
    visualize(best_out, color=data.y)



if __name__ == "__main__":
    main()
    

        

