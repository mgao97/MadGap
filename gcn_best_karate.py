import os
import copy
from tqdm import tqdm
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import KarateClub
from torch_geometric.transforms import NormalizeFeatures

import dgl



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
    plt.savefig('figs/karate_best.png')


def accuracy(y_pred, y_true):
    
    correct = y_pred.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)




def get_pos_neg_distances(data, out, hop_neigh, hop_remote):

    graph = to_networkx(data)
    print(graph)
    print(len(graph.nodes()), len(graph.edges()))

    pos_sim, neg_sim = [],[]
    for node in range(data.num_nodes):
        # Get 3-hop subgraph
        pos_nodes, _, _, _ = k_hop_subgraph(node, hop_neigh, data.edge_index)

        # pos_x = out[pos_nodes]

        for pos in pos_nodes:
            pos_cos_sim = F.cosine_similarity(out[node], out[pos], dim=0, eps=1e-8)
            pos_sim.append(pos_cos_sim)
        
        # Get 6-hop subgraph
        neg_nodes, _, _, _ = k_hop_subgraph(node, hop_remote, data.edge_index)
        # neg_x = out[neg_nodes]

        for neg in neg_nodes:
            neg_cos_sim = F.cosine_similarity(out[node], out[neg], dim=0, eps=1e-8)
            neg_sim.append(neg_cos_sim)
        
    # 使用 torch.stack() 堆叠成一个新的张量
    pos_sim_stacked = torch.stack(pos_sim, dim=0)
    neg_sim_stacked = torch.stack(neg_sim, dim=0)

    # print('pos sim shape and neg sim shape: ')
    # print(pos_sim_stacked.shape, neg_sim_stacked.shape)

    pos_dis = 1 - pos_sim_stacked
    neg_dis = 1 - neg_sim_stacked

    divide_pos = (pos_dis!=0).sum(0).type(torch.FloatTensor) + 1e-8
    divide_neg = (neg_dis!=0).sum(0).type(torch.FloatTensor) + 1e-8

    pos_dis_mean = pos_dis.sum(0) / divide_pos
    neg_dis_mean = neg_dis.sum(0) / divide_neg

    return pos_sim_stacked, neg_sim_stacked, pos_dis_mean, neg_dis_mean



class MarginRankignLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginRankignLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_sim_stacked, neg_sim_stacked, positive_distances, negative_distances):
        positive_distances = torch.clamp(positive_distances, min=0.0)  # Ensure non-negativity
        negative_distances = torch.clamp(negative_distances, min=0.0)  # Ensure non-negativity

        # Calculate the maximum positive distance
        max_positive_distance, _ = torch.max(positive_distances, dim=0)
        # Calculate the minimum negative distance
        min_negative_distance, _ = torch.min(negative_distances, dim=0)

        # Calculate the loss components
        loss_pos = torch.max(torch.zeros(1), self.margin + torch.mean(pos_sim_stacked))
        loss_neg = torch.mean(neg_sim_stacked)
        loss_margin = torch.mean(torch.clamp(self.margin + min_negative_distance - max_positive_distance, min=0))

        # Calculate the final loss
        loss = loss_pos - loss_neg + loss_margin

        return loss



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
    
    dataset = KarateClub(transform=NormalizeFeatures())
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

    karate_club_dataset = dgl.data.KarateClubDataset()
    ground_truth = karate_club_dataset[0].ndata["label"]
    #  # 创建一个空的 NetworkX 图
    # # 创建一个空的 NetworkX 图
    # G = nx.Graph()
    
    # # 添加节点
    # G.add_nodes_from(range(data.num_nodes))
    
    # # 添加边
    # edge_index = data.edge_index
    # for i in range(edge_index.shape[1]):
    #     src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    #     src, dst = edge_index[1, i].item(), edge_index[0, i].item()
    #     G.add_edge(src, dst)

    # print(f'graph diameter:', nx.diameter(G))


    model = GCN(dataset.num_features, 5, 2)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    margin_ranking = MarginRankignLoss(margin=1.0)
    hop_neigh, hop_remote = 2, 4
    best_val_acc = 0
    num_epochs = 8
    train_losses, val_losses = [], []

    test_mask = torch.tensor([ True, False, False, False,  
                               True, False, False, False,  
                               True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
                               True, False,  False, False, False, False, False, False, False, False])

    val_mask = torch.tensor([ False, True, False, False,  
                             False, True, False, False,  
                             False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
                             False, True, False, False, False, False, False, False, False, False])

    train_mask = torch.tensor([ False, False, True, True, 
                              False, False,  True, True, 
                              False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                              False, False, True, True, True, True, True, True, True, True])
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask


    
    #----------------model training----------------
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis = get_pos_neg_distances(data, out, hop_neigh, hop_remote)
        loss = cross_entropy(out[data.train_mask], ground_truth[data.train_mask]) + margin_ranking(pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis)
        
        print('*'*50)
        print(cross_entropy(out[data.train_mask], ground_truth[data.train_mask]), margin_ranking(pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis))
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)
            val_loss = cross_entropy(out[data.val_mask], ground_truth[data.val_mask]) + margin_ranking(pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis)
            
            print('='*50)
            print(cross_entropy(out[data.train_mask], ground_truth[data.train_mask]), margin_ranking(pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis))
            val_pred = F.log_softmax(out, 1).argmax(dim=-1)
            val_losses.append(val_loss.item())
            
            val_acc = accuracy(val_pred[data.val_mask], ground_truth[data.val_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        # if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val ACC: {val_acc:.4f}')


    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figs/loss_curve_karate_best.png')

    model.eval()
    # model = best_model
    out = model(data.x, data.edge_index)
    test_pred = F.log_softmax(out,1).argmax(dim=-1)
    test_acc = accuracy(test_pred[data.test_mask], ground_truth[data.test_mask])
    print(f'Test Accuracy: {test_acc:.4f}')
    
    best_out = model(data.x, data.edge_index)
    visualize(best_out, color=ground_truth)



if __name__ == "__main__":
    main()
    

        

