import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, num_hidden=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class NodeAttention(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_heads, dropout):
        super(NodeAttention, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class NET(nn.Module):
    def __init__(self,
                 meta_paths,
                 in_size,
                 n_class = None,
                 num_hidden = 16,
                 num_layers = 2,
                 heads = 8,
                 dropout = 0.6):
        super(NET, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(NodeAttention(meta_paths, in_size, num_hidden, heads, dropout))
        for l in range(1, num_layers):
            self.layers.append(NodeAttention(meta_paths, num_hidden * heads,
                                        num_hidden, heads, dropout))
        self.predict = None
        if n_class is not None:
            self.predict = torch.nn.Linear(num_hidden * heads, n_class)

    def forward(self, g, h, mini_batch = False, edge_weight = None):
        for gnn in self.layers:
            h = gnn(g, h)
        return h if self.predict is None else self.predict(h)

    def get_embeddings(self, g, h, mini_batch = False, get_attention = False):
        attentions = []
        for gnn in self.layers:
            if get_attention:
                h, a = gnn(g, h, True)
                attentions.append(a)
            else:
                h = gnn(g, h)
        return h, attentions if get_attention else h