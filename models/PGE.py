import torch
import torch.nn.functional as F
from itertools import product
import numpy as np
import dgl

class PGE(torch.nn.Module):

    def __init__(self, in_feat, num_node, hidden=128, num_layer=3, device=None):
        super(PGE, self).__init__()

        self.layers = torch.nn.ModuleList([])
        self.layers.append(torch.nn.Linear(in_feat*2, hidden))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden))
        for i in range(num_layer-2):
            self.layers.append(torch.nn.Linear(hidden, hidden))
            self.bns.append(torch.nn.BatchNorm1d(hidden))
        self.layers.append(torch.nn.Linear(hidden, 1))

        edge_index = np.array(list(product(range(num_node), range(num_node))))
        self.edge_index = edge_index.T
        self.num_node = num_node
        self.device = device
        self.reset_parameters()

    def forward(self, x):
        edge_index = self.edge_index
        edge_embed = torch.cat([x[edge_index[0]],
                x[edge_index[1]]], axis=1)
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.num_node, self.num_node)

        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))

        g_selfloop = adj + torch.eye(adj.shape[0]).to(self.device)
        edge_index = g_selfloop.nonzero().T
        g = dgl.graph((edge_index[0],edge_index[1])).to(self.device)
        edge_weight = g_selfloop[edge_index[0],edge_index[1]]
        return g, edge_weight

    @torch.no_grad()
    def inference(self, x):
        g, edge_weight = self.forward(x)
        return g, edge_weight

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
            if isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)