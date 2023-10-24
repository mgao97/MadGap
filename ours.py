import os
import copy
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch_geometric.utils as pyg_utils


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
    plt.savefig('figs/cora_ours.png')


def accuracy(y_pred, y_true):
    
    correct = y_pred.eq(y_true).double()
    correct = correct.sum().item()

    accuracy_value = correct / len(y_true)
    # 计算准确率的标准差
    squared_error = (correct - accuracy_value * len(y_true))**2
    variance = squared_error / len(y_true)
    accuracy_std = np.sqrt(variance)

    return accuracy_value, accuracy_std

def compute_sparsity_and_entropy(features):
    # 计算特征的稀疏性
    sparsity = 1.0 - (torch.sum(features != 0, dim=1, dtype=torch.float32) / features.shape[1])

    # 计算特征的信息熵
    def entropy(arr):
        values, counts = np.unique(arr, return_counts=True)
        prob = counts / counts.sum()
        return -np.sum(prob * np.log(prob))

    entropy_values = np.array([entropy(features[i].numpy()) for i in range(features.shape[0])])
    entropy_values = torch.tensor(entropy_values, dtype=torch.float32)

    return sparsity, entropy_values/len(np.unique(features))

def compute_edge_weight(data):
    # 计算邻接节点的度值
    row, col = data.edge_index
    deg = pyg_utils.degree(col, data.num_nodes)
    
    # 计算两个节点之间的距离（假设有节点距离信息）
    # 请根据您的实际情况获取节点之间的距离信息

    # 计算权重
    distance_weight = 1.0 
    
    # 此处的 distance_weight 是您根据距离计算的权重，可以根据需要进一步调整

    # 使用度值和权重计算最终的边权重
    edge_weight = deg[row].float() * deg[col].float() * distance_weight
    
    # 对权重进行归一化，以便在消息传递中使用
    edge_weight = F.normalize(edge_weight, p=1, dim=0)

    return edge_weight



# Define a custom GCN layer with feature sparsity and entropy as neighbor weights
class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lambda_sparsity = nn.Parameter(torch.tensor(0.01))  # Initialize with a default value
        self.lambda_entropy = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.lambda_sparsity, 0.01)  # 例如，将 lambda_sparsity 初始化为 0.1
        nn.init.constant_(self.lambda_entropy, 0.1) 
        nn.init.constant_(self.beta, 0.5) 

    def forward(self, x, edge_index, edge_weight):
        # Calculate the sparsity of node features
        feature_sparsity = 1.0 - (torch.sum(x != 0, dim=1, dtype=torch.float32) / x.shape[1])
        
        # Calculate the entropy of node features
        feature_entropy = -torch.sum(x * torch.log(x + 1e-15), dim=1)
        
        # Normalize feature sparsity and entropy
        feature_sparsity = (feature_sparsity - feature_sparsity.min()) / (feature_sparsity.max() - feature_sparsity.min())
        feature_entropy = (feature_entropy - feature_entropy.min()) / (feature_entropy.max() - feature_entropy.min())
        
        # Compute neighbor weights based on feature sparsity and entropy
        neighbor_weights = (feature_sparsity / feature_entropy) * 0.001
        
        # Normalize neighbor weights
        neighbor_weights = F.normalize(neighbor_weights, p=1, dim=0)
        # Expand neighbor weights to edge weights
        expanded_neighbor_weights = neighbor_weights[edge_index[0]]
        
        # Update node features with sparsity and entropy information
        x = torch.cat((x, feature_sparsity.view(-1, 1), feature_entropy.view(-1, 1)), dim=1)
        # data.x = x
        return self.propagate(edge_index, x=x, edge_weight=edge_weight * expanded_neighbor_weights)


    def message(self, x_j, edge_weight):
        return x_j

    def update(self, aggr_out):
        return torch.matmul(aggr_out, self.weight)

# Define a GCN model
class GCNModel(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GCNModel, self).__init__()
        self.lambda_sparsity = nn.Parameter(torch.tensor(0.01))  # Initialize with a default value
        self.lambda_entropy = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.conv1 = CustomGCNConv(num_features + 2, num_hidden)  # Add 2 for feature sparsity and entropy
        self.conv2 = CustomGCNConv(num_hidden+2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = compute_edge_weight(data)
        x = self.conv1(x, edge_index, edge_weight)
        # print(x.shape)
        # x, edge_index = data.x, data.edge_index
        edge_weight = compute_edge_weight(data)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)





def main():
    
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    data.edge_weight = compute_edge_weight(data)
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

    
    num_hidden=512
    # Define additional regularization hyperparameters
    
    
    

    # Initialize the GCN model
    model = GCNModel(data.num_features, num_hidden, dataset.num_classes)
    print(model)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)

    # Training loop
    def train(best_val_acc, train_losses, val_losses):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss_classification = criterion(out[data.train_mask], data.y[data.train_mask])

        # Calculate feature sparsity and entropy loss
        feature_sparsity_loss = torch.mean(data.x[data.train_mask, -2])
        feature_entropy_loss = -torch.mean(data.x[data.train_mask, -1])

        # Combine classification loss with regularization terms
        total_loss = model.beta * loss_classification - model.lambda_sparsity * feature_sparsity_loss + model.lambda_entropy * feature_entropy_loss
        train_losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(data)
            val_loss_classification = criterion(out[data.val_mask], data.y[data.val_mask])
            # Calculate feature sparsity and entropy loss
            val_feature_sparsity_loss = torch.mean(data.x[data.val_mask, -2])
            val_feature_entropy_loss = -torch.mean(data.x[data.val_mask, -1])

            # Combine classification loss with regularization terms
            val_total_loss = model.beta * val_loss_classification - model.lambda_sparsity * val_feature_sparsity_loss + model.lambda_entropy * val_feature_entropy_loss
            val_losses.append(val_total_loss.item())
            val_pred = out.argmax(dim=1)
            val_acc,_ = accuracy(val_pred[data.val_mask], data.y[data.val_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), 'models/ours.pth')

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss:.4f}, Val Loss: {val_total_loss:.4f}, Val ACC: {val_acc:.4f}')

        return train_losses, val_losses

    def test():
        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits[data.test_mask].max(1)[1]
            correct = pred.eq(data.y[data.test_mask]).sum().item()
            total = data.test_mask.sum().item()
            accuracy = correct / total

            # 计算准确率的标准差
            squared_error = (correct - accuracy * len(data.y[data.test_mask]))**2
            variance = squared_error / len(data.y[data.test_mask])
            accuracy_std = np.sqrt(variance)
            return accuracy, accuracy_std
    
    
    best_val_acc = 0
    num_epochs = 140
    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs)):
        train_losses, val_losses = train(best_val_acc, train_losses, val_losses)

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figs/loss_ours.png')

    model = GCNModel(data.num_features, num_hidden, dataset.num_classes)
    model.load_state_dict(torch.load('models/ours.pth'))
    test_acc, test_acc_std = test()
    print(f'Test Accuracy: {test_acc:.4f}, Test Acc std: {test_acc_std:.4f}')

    # model.eval()
    best_out = model(data)
    visualize(best_out, color=data.y)



if __name__ == "__main__":
    main()
    

        

