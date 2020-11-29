import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric import datasets

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hiden, classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(feature, hiden)
        self.conv2 = SAGEConv(hiden, classes)

    def forward(self, input, adj):
        x, edge_index = input, adj
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x,dim=1)