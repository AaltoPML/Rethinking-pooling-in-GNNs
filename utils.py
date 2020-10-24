import numpy as np
import torch
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
import copy
import torch.nn.functional as F

EPS = 1e-15


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_assign_matrix(random, dim1, dim2, normalize=False):
    if random == 'uniform':
        m = torch.rand(dim1, dim2)
    elif random == 'normal':
        m = torch.randn(dim1, dim2)
    elif random == 'bernoulli':
        m = torch.bernoulli(0.3*torch.ones(dim1, dim2))
    elif random == 'categorical':
        idxs = torch.multinomial((1.0/dim2)*torch.ones((dim1, dim2)), 1)
        m = torch.zeros(dim1, dim2)
        m[torch.arange(dim1), idxs.view(-1)] = 1.0

    if normalize:
        m = m / (m.sum(dim=1, keepdim=True) + EPS)
    return m


def knn_filter(data, k=8):
    dists = torch.cdist(data.pos, data.pos)
    sorted_dists, _ = dists.sort(dim=-1)
    sigmas = sorted_dists[:, 1:k + 1].mean(dim=-1) + 1e-8  # avoid division by 0
    adj = (-(dists / sigmas) ** 2).exp()  # adjacency matrix
    adj = 0.5 * (adj + adj.T)  # gets a symmetric matrix
    adj.fill_diagonal_(0)  # removing self-loops
    knn_values, knn_ids = adj.sort(dim=-1, descending=True)
    adj[torch.arange(adj.size(0)).unsqueeze(1), knn_ids[:, k:]] = 0.0  # selecting the knns. The matrix is not symmetric.
    data.edge_index = adj.nonzero().T
    data.edge_attr = adj[adj > 0].view(-1).unsqueeze(1)
    data.x = torch.cat((data.x, data.pos), dim=1)
    return data


def rwr_filter(data, c=0.1):
    adj = to_dense_adj(data.edge_index).squeeze()
    adj = 0.5*(adj + adj.T)
    adj = adj + torch.eye(adj.shape[0])
    d = torch.diag(torch.sum(adj, 1))
    d_inv = d**(-0.5)
    d_inv[torch.isinf(d_inv)] = 0.
    w_tilda = torch.matmul(d_inv, adj)
    w_tilda = np.matmul(w_tilda, d_inv)
    q = torch.eye(w_tilda.shape[0]) - c * w_tilda
    q_inv = torch.inverse(q)
    rwr = (1 - c) * q_inv
    rwr, _ = torch.sort(rwr, dim=1, descending=True)
    sparse_rwr = rwr.to_sparse()
    data.edge_index = sparse_rwr.indices()
    data.edge_attr = sparse_rwr.values().unsqueeze(1).float()
    return data


def graph_permutation(data):
    d2 = copy.deepcopy(data)
    for i in range(d2.batch.max()+1):
        idx = (d2.batch == i).nonzero().squeeze()
        gsize = idx.shape[0]
        rp = torch.randperm(idx.shape[0])
        idx_perm = idx[rp]
        d2.x[idx_perm, :] = data.x[idx, :]
        for j in range(gsize):
            d2.edge_index[data.edge_index == idx[j]] = idx_perm[j]
    return d2


# code taken from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding) + F.relu(x + self.root_emb.weight)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
