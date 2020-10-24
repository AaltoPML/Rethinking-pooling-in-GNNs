from math import ceil

import torch
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import DenseGraphConv
from utils import fetch_assign_matrix, GCNConv

NUM_SAGE_LAYERS = 3


class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = nn.Linear((NUM_SAGE_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            # GNN's intermediate representation is given by the concatenation of SAGE layers
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, add_loop=False)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, add_loop=False)))
        x3 = self.conv3(x2, adj, mask, add_loop=False)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x


class DiffPoolLayer(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_embedding, current_num_clusters,
                 no_new_clusters, pooling_type, invariant):

        super().__init__()
        self.pooling_type = pooling_type
        self.invariant = invariant
        if pooling_type != 'gnn':
            if self.invariant:
                self.rm = fetch_assign_matrix(pooling_type, dim_input, no_new_clusters)
            else:
                self.rm = fetch_assign_matrix(pooling_type, current_num_clusters, no_new_clusters)
            self.rm.requires_grad = False
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):

        if self.pooling_type == 'gnn':
            s = self.gnn_pool(x, adj, mask)
        else:
            if self.invariant:
                s = self.rm.unsqueeze(dim=0).expand(x.size(0), -1, -1)
                s = s.to(x.device)
                s = x.detach().matmul(s)
            else:
                s = self.rm[:x.size(1), :].unsqueeze(dim=0)
                s = s.expand(x.size(0), -1, -1)
                s = s.to(x.device)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPool(nn.Module):

    def __init__(self, num_features, num_classes, max_num_nodes, num_layers, gnn_hidden_dim,
                 gnn_output_dim, mlp_hidden_dim, pooling_type, invariant, encode_edge, pre_sum_aggr=False):
        super().__init__()

        self.encode_edge = encode_edge
        self.pre_sum_aggr = pre_sum_aggr
        self.max_num_nodes = max_num_nodes
        self.pooling_type = pooling_type
        self.num_diffpool_layers = num_layers

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_layers == 1 else 0.25

        gnn_dim_input = num_features
        if encode_edge:
            gnn_dim_input = gnn_hidden_dim
            self.conv1 = GCNConv(gnn_hidden_dim, aggr='add')

        if self.pre_sum_aggr:
            self.conv1 = DenseGraphConv(gnn_dim_input, gnn_dim_input)

        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        gnn_embed_dim_output = (NUM_SAGE_LAYERS - 1) * gnn_hidden_dim + gnn_output_dim

        layers = []
        current_num_clusters = self.max_num_nodes
        for i in range(num_layers):

            diffpool_layer = DiffPoolLayer(gnn_dim_input, gnn_hidden_dim, gnn_output_dim, current_num_clusters,
                                           no_new_clusters, pooling_type, invariant)
            layers.append(diffpool_layer)

            # Update embedding sizes
            gnn_dim_input = gnn_embed_dim_output
            current_num_clusters = no_new_clusters
            no_new_clusters = ceil(no_new_clusters * coarse_factor)

        self.diffpool_layers = nn.ModuleList(layers)

        # After DiffPool layers, apply again layers of GraphSAGE convolutions
        self.final_embed = SAGEConvolutions(gnn_embed_dim_output, gnn_hidden_dim, gnn_output_dim, lin=False)
        final_embed_dim_output = gnn_embed_dim_output * (num_layers + 1)

        self.lin1 = nn.Linear(final_embed_dim_output, mlp_hidden_dim)
        self.lin2 = nn.Linear(mlp_hidden_dim, num_classes)
        self.atom_encoder = AtomEncoder(emb_dim=gnn_hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = self.conv1(x, edge_index, data.edge_attr)

        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)

        if self.pre_sum_aggr:
            x = self.conv1(x, adj, mask)

        x_all, l_total, e_total = [], 0, 0

        for i in range(self.num_diffpool_layers):
            if i != 0:
                mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])

            l_total += l
            e_total += e

        x = self.final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diffpool layers)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, l_total, e_total
