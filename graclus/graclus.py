import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import GraphConv, graclus, max_pool, global_mean_pool, JumpingKnowledge
from graclus.negative_edges import batched_negative_edges
from ogb.graphproppred.mol_encoder import AtomEncoder
from utils import GCNConv


class Graclus(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden, pooling_type,
                 no_cat=False, encode_edge=False):
        super(Graclus, self).__init__()
        self.encode_edge = encode_edge
        if encode_edge:
            self.conv1 = GCNConv(hidden, aggr='add')
        else:
            self.conv1 = GraphConv(num_features, hidden, aggr='add')

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden, aggr='add'))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        if no_cat:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.pooling_type = pooling_type
        self.no_cat = no_cat

        self.atom_encoder = AtomEncoder(emb_dim=hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = self.conv1(x, edge_index, data.edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if self.pooling_type != 'none':
                if self.pooling_type == 'complement':
                    complement = batched_negative_edges(edge_index=edge_index, batch=batch, force_undirected=True)
                    cluster = graclus(complement, num_nodes=x.size(0))
                elif self.pooling_type == 'graclus':
                    cluster = graclus(edge_index, num_nodes=x.size(0))
                data = Batch(x=x, edge_index=edge_index, batch=batch)
                data = max_pool(cluster, data)
                x, edge_index, batch = data.x, data.edge_index, data.batch

        if not self.no_cat:
            x = self.jump(xs)
        else:
            x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
