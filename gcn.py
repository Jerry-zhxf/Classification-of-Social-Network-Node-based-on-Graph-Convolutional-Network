import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric import datasets

class GraphConvolution(torch.nn.Module):
    """GCN layer"""

    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features,out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features,self.out_features,self.bias is not None
        )

class GCN(torch.nn.Module):
    def __init__(self, feature, hiden, classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature, hiden)
        self.conv2 = GCNConv(hiden, classes)

    def forward(self, input, adj):
        x, edge_index = input, adj
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x,dim=1)