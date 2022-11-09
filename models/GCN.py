import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class NET(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 activation = F.relu,
                 dropout = 0.5,
                 n_layers = 2,
                 n_classes = None):
        super(NET, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.predict = None
        if n_classes is not None:
            self.predict = torch.nn.Linear(n_hidden, n_classes)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

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
