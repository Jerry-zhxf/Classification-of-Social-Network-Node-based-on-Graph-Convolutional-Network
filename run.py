import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np
from gcn import GCN
from graphsage import GraphSAGE
from utils import visualize_graph_res
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='GCN',help='model to be trained')
parser.add_argument('--dataset', type=str, default='Cora', help='dataset running with.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset = Planetoid(root='/tmp/'+args.dataset,name=args.dataset)
data = dataset[0].to(device)

# generate model and optimizer with parameter
if args.model =='GCN':
    model = GCN(dataset.num_features, args.hidden, dataset.num_classes).to(device)
else:
    model = GraphSAGE(dataset.num_features, args.hidden, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# define two list for plot
Accuracy_list = []
Loss_list = []

# train the model
model.train()
for epoch in range(args.epochs):
    optimizer.zero_grad()
    out = model(data.x,data.edge_index)
    _, pred = model(data.x,data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # statistics acc and loss of every iteration
    Accuracy_list.append(acc)
    Loss_list.append(loss)
    loss.backward()
    optimizer.step()

# evaluate the model
model.eval()
_,pred = model(data.x,data.edge_index).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print(args.model+' Accuracy: {:.4f}'.format(acc))

# plot training figure
x = range(0, args.epochs)
y1 = Accuracy_list
y2 = Loss_list
fig = plt.figure()

# set y-axis 1 for accuracy
ax1 = fig.add_subplot(111)
ax1.plot(x,y1,label='Accuracy')
ax1.set_ylabel('Test accuracy')
ax1.legend(loc=1)

# set y-axis 2 for loss
ax2 = ax1.twinx()
ax2.set_ylabel('Test loss')
ax2.plot(x,y2,'r',label='Loss')
ax2.legend(loc=2)

ax1.set_xlabel('Epoches')
plt.show()
#plt.savefig("cora.jpg")

#visualize the result

h_data = data.x
h_data = h_data.to('cpu')
embedding_weight_before = h_data.detach().numpy()

out = model(data.x, data.edge_index)
out = out.to('cpu')
embedding_weight_after = out.detach().numpy()

label = data.y
label = label.to('cpu')
label = label.detach().numpy()

node_pos1, color_idx1 = visualize_graph_res(embedding_weight_before, np.arange(data.num_nodes), label)
node_pos2, color_idx2 = visualize_graph_res(embedding_weight_after, np.arange(data.num_nodes), label)

plt.figure(num=2, figsize=(800,600))
#plt.canvas.manager.full_screen_toggle()
ax1 = plt.subplot(121)
for c, idx in color_idx1.items():
    ax1.scatter(node_pos1[idx, 0], node_pos1[idx, 1], label=c)

ax1.set_title('Before Training')


ax2 = plt.subplot(122)
for c, idx in color_idx2.items():
    ax2.scatter(node_pos2[idx, 0], node_pos2[idx, 1], label=c)
ax2.set_title('after Training')
plt.show()

