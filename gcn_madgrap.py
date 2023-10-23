import os
import copy
from tqdm import tqdm
import torch
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
    plt.savefig('figs/cora_best.png')


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[0].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


#the tensor version for mad_gap (Be able to transfer gradients)
#intensor: [node_num * hidden_dim], the node feature matrix;
#neb_mask,rmt_mask:[node_num * node_num], the mask matrices of the neighbor and remote raltion;
#target_idx = [1,2,3...n], the nodes idx for which we calculate the mad_gap value;
def mad_gap_regularizer(intensor,neb_mask,rmt_mask,target_idx):
    node_num,feat_num = intensor.size()

    input1 = intensor.expand(node_num,node_num,feat_num)
    input2 = input1.transpose(0,1)

    input1 = input1.contiguous().view(-1,feat_num)
    input2 = input2.contiguous().view(-1,feat_num)

    simi_tensor = F.cosine_similarity(input1,input2, dim=1, eps=1e-8).view(node_num,node_num)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor,neb_mask)
    rmt_dist = torch.mul(dist_tensor,rmt_mask)
    
    divide_neb = (neb_dist!=0).sum(1).type(torch.FloatTensor).cuda() + 1e-8
    divide_rmt = (rmt_dist!=0).sum(1).type(torch.FloatTensor).cuda() + 1e-8

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = torch.mean(neb_mean_list[target_idx])
    rmt_mad = torch.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap



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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    margin_ranking = MarginRankignLoss(margin=1.0)
    hop_neigh, hop_remote = 3, 8
    best_val_acc = 0
    num_epochs = 1500
    
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis = get_pos_neg_distances(data, out, hop_neigh, hop_remote)
        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])+margin_ranking(pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)
            val_loss = cross_entropy(out[data.val_mask], data.y[data.val_mask])+margin_ranking(pos_sim_stacked, neg_sim_stacked, pos_dis, neg_dis)
            val_pred = out.argmax(dim=1)
            
            val_acc = accuracy(val_pred[data.val_mask], data.y[data.val_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        return loss, val_loss, val_acc, best_model
    

    def test():
        best_model.eval()
        out = best_model(data.x, data.edge_index)
        test_pred = out.argmax(dim=1)
        test_acc = accuracy(test_pred[data.test_mask], data.y[data.test_mask])

        return test_acc


    for epoch in tqdm(range(1, num_epochs+1)):
        loss, val_loss, val_acc, best_model = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')

        # Plot the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('figs/loss_curve_best.png')

        test_acc = test()
        print(f'Test Accuracy: {test_acc:.4f}')

    best_model.eval()
    best_out = best_model(data.x, data.edge_index)
    visualize(best_out, color=data.y)



if __name__ == "__main__":
    main()
    

        

