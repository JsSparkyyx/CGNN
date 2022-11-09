"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from .GATConv import GATConv
import torch.nn.functional as F

class NET(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 heads = 8,
                 num_layers = 2,
                 n_class = None,
                 activation = None,
                 feat_drop = 0.6,
                 attn_drop = 0.6,
                 negative_slope = 0.2,
                 residual = True):
        super(NET, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads, num_hidden, heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        self.predict = None
        if n_class is not None:
            self.predict = torch.nn.Linear(num_hidden * heads, n_class)
            # self.predict = GATConv(
            #     num_hidden * heads, n_class, 1,
            #     feat_drop, attn_drop, negative_slope, residual, None)

    def forward(self, g, inputs, mini_batch = False, edge_weight = None):
        h = inputs
        if not mini_batch:
            for l in range(self.num_layers):
                h = self.gat_layers[l](g, h, edge_weight = edge_weight).flatten(1)
        else:
            for block, gat_layer in zip(g, self.gat_layers):
                h_dst = h[block.dstnodes()]
                h = gat_layer(block, (h,h_dst), edge_weight = edge_weight).flatten(1)
        return h if self.predict is None else self.predict(h)
        # print(self.predict(g, h, edge_weight = edge_weight))
        # return h if self.predict is None else self.predict(g, h, edge_weight = edge_weight).squeeze(1)
