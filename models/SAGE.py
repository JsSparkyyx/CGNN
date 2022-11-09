import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .SAGEConv import SAGEConv

class NET(nn.Module):
    def __init__(self,
                in_feats,
                n_hidden,
                activation = F.relu,
                dropout = 0.5,
                aggregator_type = 'mean',
                n_layers = 2,
                n_classes = None):
        super(NET, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(1, n_layers):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.predict = None
        if n_classes is not None:
            self.predict = torch.nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, inputs, mini_batch = False, edge_weight = None):
        h = inputs
        if not mini_batch:
            for l in range(self.n_layers):
                h = self.layers[l](g, h, edge_weight = edge_weight)
                h = self.activation(h)
                h = self.dropout(h)
        else:
            for block, layer in zip(g, self.layers):
                h_dst = h[block.dstnodes()]
                h = layer(block, (h,h_dst), edge_weight = edge_weight)
                h = self.activation(h)
                h = self.dropout(h)
        return h if self.predict is None else self.predict(h)