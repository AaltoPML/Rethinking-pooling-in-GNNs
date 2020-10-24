from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import DenseGraphConv, GraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from mincut.mincut_pool_mod import dense_mincut_pool
from utils import fetch_assign_matrix
from ogb.graphproppred.mol_encoder import AtomEncoder
from utils import GCNConv


class MincutPool(torch.nn.Module):
    def __init__(self, num_features, num_classes, max_num_nodes, hidden, pooling_type,
                 num_layers, encode_edge=False):
        super(MincutPool, self).__init__()
        self.encode_edge = encode_edge

        self.atom_encoder = AtomEncoder(emb_dim=hidden)

        self.pooling_type = pooling_type
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            if i == 0:
                if encode_edge:
                    self.convs.append(GCNConv(hidden, aggr='add'))
                else:
                    self.convs.append(GraphConv(num_features, hidden, aggr='add'))
            else:
                self.convs.append(DenseGraphConv(hidden, hidden))

        self.rms = []
        num_nodes = max_num_nodes
        for i in range(num_layers - 1):
            num_nodes = ceil(0.5 * num_nodes)
            if pooling_type == 'mlp':
                self.pools.append(Linear(hidden, num_nodes))
            else:
                self.rms.append(fetch_assign_matrix('uniform', ceil(2 * num_nodes), num_nodes))

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = F.relu(self.convs[0](x, edge_index, data.edge_attr))
        else:
            x = F.relu(self.convs[0](x, edge_index))

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        if self.pooling_type != 'mlp':
            s = self.rms[0][:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
        else:
            s = self.pools[0](x)

        x, adj, mc, o = dense_mincut_pool(x, adj, s, mask)

        for i in range(1, self.num_layers - 1):
            x = F.relu(self.convs[i](x, adj))
            if self.pooling_type != 'mlp':
                s = self.rms[i][:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
            else:
                s = self.pools[i](x)
            x, adj, mc_aux, o_aux = dense_mincut_pool(x, adj, s)
            mc += mc_aux
            o += o_aux

        x = self.convs[self.num_layers-1](x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, mc, o
