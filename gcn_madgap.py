import os
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph, mask_to_index, subgraph, to_undirected
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
    plt.savefig('figs/cora_madgap.png')


def accuracy(y_pred, y_true):
    
    correct = y_pred.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def compute_2hop_neighbor_matrix(data, target_index):
    # 创建一个零矩阵，形状为(num_nodes, num_nodes)
    num_nodes = data.num_nodes
    neighbor_matrix = torch.zeros((num_nodes, num_nodes))

    # 获取目标节点的邻居节点
    target_neighbors = set(data.edge_index[1, data.edge_index[0] == target_index].tolist())

    # 创建一个副本以避免修改正在迭代的集合
    new_target_neighbors = target_neighbors.copy()

    # 获取2跳邻居节点
    for neighbor in target_neighbors:
        neighbor_neighbors = set(data.edge_index[1, data.edge_index[0] == neighbor].tolist())
        new_target_neighbors.update(neighbor_neighbors)

    return neighbor_matrix



def compute_8hop_neighbor_matrix(data, target_index, hop_remote):
    # 创建一个零矩阵，形状为(num_nodes, num_nodes)
    num_nodes = data.num_nodes
    neighbor_matrix = torch.zeros((num_nodes, num_nodes))

    # 获取目标节点的邻居节点
    target_neighbors = set(data.edge_index[1, data.edge_index[0] == target_index].tolist())

    # 逐层获取邻居节点，直到8跳
    for _ in range(hop_remote):
        new_neighbors = set()
        for neighbor in target_neighbors:
            new_neighbors.update(data.edge_index[1, data.edge_index[0] == neighbor].tolist())
        target_neighbors.update(new_neighbors)

    # 填充1，表示目标节点和8-hop之外的邻居节点之间有连接
    neighbor_matrix[target_index, list(target_neighbors)] = 1
    neighbor_matrix[list(target_neighbors), target_index] = 1

    # 将8-hop以内的邻接信息标注为0
    neighbor_matrix[target_index, target_index] = 0
    neighbor_matrix[list(target_neighbors), list(target_neighbors)] = 0

    return neighbor_matrix



#the tensor version for mad_gap (Be able to transfer gradients)
#intensor: [node_num * hidden_dim], the node feature matrix;
#neb_mask,rmt_mask:[node_num * node_num], the mask matrices of the neighbor and remote raltion;
#target_idx = [1,2,3...n], the nodes idx for which we calculate the mad_gap value;
def mad_gap_regularizer(intensor, neb_mask, rmt_mask, target_idx):
    node_num,feat_num = intensor.size()

    input1 = intensor.expand(node_num,node_num,feat_num)
    input2 = input1.transpose(0,1)

    input1 = input1.contiguous().view(-1,feat_num)
    input2 = input2.contiguous().view(-1,feat_num)

    simi_tensor = F.cosine_similarity(input1,input2, dim=1, eps=1e-8).view(node_num,node_num)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor,neb_mask)
    rmt_dist = torch.mul(dist_tensor,rmt_mask)
    
    divide_neb = (neb_dist!=0).sum(1).type(torch.FloatTensor) + 1e-8
    divide_rmt = (rmt_dist!=0).sum(1).type(torch.FloatTensor) + 1e-8

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = torch.mean(neb_mean_list[target_idx])
    rmt_mad = torch.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha):
        super().__init__()
        torch.manual_seed(42)
        self.alpha = alpha
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.conv2(x, edge_index)
        return x, out


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


    model = GCN(dataset.num_features, 512, dataset.num_classes, alpha=0.5)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    num_epochs = 5
    madgap_train, madgap_val = [], []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        emb, out = model(data.x, data.edge_index)
        for target_index in range(data.num_nodes):
            target_index = torch.tensor(target_index)
            neb_mask = compute_2hop_neighbor_matrix(data, target_index)
            rmt_mask = compute_8hop_neighbor_matrix(data, target_index, hop_remote=8)
            madgap = mad_gap_regularizer(emb, neb_mask, rmt_mask, target_index)
            # print(neb_mask.shape, rmt_mask.shape, madgap, madgap.shape)
            madgap_train.append(madgap)

        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask]) - model.alpha * torch.stack(madgap_train)
        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            model.eval()
            emb, out = model(data.x, data.edge_index)
            for target_index in range(data.num_nodes):
                target_index = torch.tensor(target_index)
                neb_mask = compute_2hop_neighbor_matrix(data, target_index)
                rmt_mask = compute_8hop_neighbor_matrix(data, target_index, hop_remote=8)
                madgap = mad_gap_regularizer(emb, neb_mask, rmt_mask, target_index)
                madgap_val.append(madgap)
            
            val_loss = cross_entropy(out[data.val_mask], data.y[data.val_mask]) - model.alpha * torch.stack(madgap_val)
            val_pred = out.argmax(dim=1)
            
            val_acc = accuracy(val_pred[data.val_mask], data.y[data.val_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figs/loss_curve_madgap.png')

    model.eval()
    model = best_model
    out = model(data.x, data.edge_index)
    test_pred = out.argmax(dim=1)
    test_acc = accuracy(test_pred[data.test_mask], data.y[data.test_mask])
    print(f'Test Accuracy: {test_acc:.4f}')






if __name__ == "__main__":
    main()
    

        

