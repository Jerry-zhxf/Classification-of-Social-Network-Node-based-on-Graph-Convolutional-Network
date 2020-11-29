import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=False, norm=True):
        super(GCNLayer, self).__init__()
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        # x A^T + b
        self.linear = nn.Linear(in_feats, out_feats)
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(p=0.5) # need more time!
        self.norm = norm

    def forward(self, g, h):
        # hat_D hat_A H W hat_D
        # g is the graph and the inputs is the input node features
        # first set the node features
        if self.dropout:
            h = self.dropout(h)
        h = self.linear(h)
        if self.norm:
            g.ndata['h'] = g.ndata['norm'] * h
        else:
            g.ndata['h'] = h
        # trigger message passing on all edges
        # Edge UDF: EdgeBatch -> dict, notice it acts on a batch of nodes
        # g.edges() indicates sending along all the edges
        g.update_all(
            message_func=fn.copy_src(src='h', out='m'),
            # {'m': edges.src['h']}
            reduce_func=fn.sum(msg='m', out='h')
            # {'h': torch.sum(nodes.mailbox['m'], dim=1)}
        )
        # get the result node features
        h = g.ndata.pop('h')
        if self.norm:
            h = h * g.ndata['norm']
        # perform linear transformation
        return h

class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size, False, False)
        self.gcn2 = GCNLayer(hidden_size, num_classes, False, False)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h