import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn import SAGEConv
import time
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset, RedditDataset
import tqdm
from NeighborSampler import NeighborSampler
import argparse
from gcn import GCN
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='Cora',help='dataset to be test')
parser.add_argument('--aggregator', type=str, default='gcn',help='aggregator to be used')
args = parser.parse_args()

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, args.aggregator))
        for i in range(1, n_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(n_hidden, n_hidden, args.aggregator))
        self.layers.append(SAGEConv(n_hidden, n_classes, args.aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(),
                         self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[start:end] = h.cpu()
            x = y
        return y

def compute_acc(pred, labels):

    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_mask, batch_size, device):

    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])

def load_subtensor(g, labels, seeds, input_nodes, device):

    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# Parameter
gpu = -1
num_epochs = 200
num_hidden = 16
num_layers = 2
fan_out = '10,25'
batch_size = 100
log_every = 20
eval_every = 10
lr = 0.001
dropout = 0.5
num_workers = 0

if gpu >= 0:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# load reddit data
# NumNodes: 232965
# NumEdges: 114848857
# NumFeats: 602
# NumClasses: 41
# NumTrainingSamples: 153431
# NumValidationSamples: 23831
# NumTestSamples: 55703\
if args.dataset =='Cora':
    data = CoraGraphDataset()
elif args.dataset=='CiteSeer':
    data = CiteseerGraphDataset()
elif args.dataset =='PubMed':
    data = PubmedGraphDataset()
elif args.dataset =='Reddit':
    data = RedditDataset(self_loop=True)
try:
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = torch.Tensor(data.features)
    in_feats = features.shape[1]
    labels = torch.LongTensor(data.labels)
    n_classes = data.num_labels

    g = data[0]
    g.ndata['features'] = features

    train_nid = torch.LongTensor(np.nonzero(train_mask)[0])
    val_nid = torch.LongTensor(np.nonzero(val_mask)[0])
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in fan_out.split(',')])


    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)


    model = GraphSAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    avg = 0
    iter_tput = []
    for epoch in range(num_epochs):
        tic = time.time()

        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time()+1 - tic_step))
            if step % log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

    eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, batch_size, device)
    print(args.dataset+'-GCN-'+args.aggregator+'-'+'Avg epoch time: {}'.format(avg / (epoch - 4)))
    print(args.dataset+'-GCN-'+args.aggregator+'-'+'Final Acc {:.4f}'.format(eval_acc))
except:
    print('dataset load error')