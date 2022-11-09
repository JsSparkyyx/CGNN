import torch
import dgl
import numpy as np

class GraphData:
    def __init__(self, g, features, labels, encoded_labels, train_mask, val_mask, test_mask, device = 'cuda:0'):
        self.device = device
        self.g = g.to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.encoded_labels = encoded_labels.to(device)
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.num_class = labels.unique().size()[0]
        self.dataloader = None
        self.class_idx = None
        self.class_sampler = None

    def retrieve_data(self):
        return self.g, self.features, self.labels, self.encoded_labels, self.train_mask, self.val_mask, self.test_mask

    def graph_sampling(self, batch_size = 256, neighbors = [10,25], shuffle = True):
        if self.dataloader is None:
            g = self.g
            train_mask = self.train_mask
            train_idx = torch.nonzero(train_mask).flatten()
            sampler = dgl.dataloading.NeighborSampler(neighbors)
            dataloader = dgl.dataloading.DataLoader(g, train_idx, sampler, batch_size = batch_size, shuffle = True)
        return dataloader

    def class_sampling(self, cls, batch_size = 256, neighbors = [10,15], shuffle = True):
        if self.class_idx is None or self.sampler is None:
            self.class_idx = {}
            for i in self.encoded_labels.unique():
                condition = torch.BoolTensor([j == i for j in self.encoded_labels])
                condition = condition * torch.BoolTensor([j == 1 for j in self.train_mask])
                mask = torch.where(condition, torch.tensor(1), torch.tensor(0)).bool()
                self.class_idx[int(i)] = torch.nonzero(mask).flatten().to(self.device)
            self.sampler = dgl.dataloading.NeighborSampler(neighbors)
        sub_g = dgl.node_subgraph(self.g,self.class_idx[cls])
        batch = torch.LongTensor(np.random.permutation(list(range(self.class_idx[cls].size()[0])))[:batch_size]).to(self.device)
        seed_nodes, output_nodes, blocks = self.sampler.sample_blocks(sub_g,batch)
        return sub_g.srcdata[dgl.NID][seed_nodes], sub_g.srcdata[dgl.NID][output_nodes], blocks