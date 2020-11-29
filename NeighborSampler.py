import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn import SAGEConv
import time
from dgl.data import RedditDataset
import tqdm

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        """
        g 为 DGLGraph；
        fanouts 为采样节点的数量，实验使用 10,25，指一阶邻居采样 10 个，二阶邻居采样 25 个。
        """
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:

            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)

            block = dgl.to_block(frontier, seeds)

            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks