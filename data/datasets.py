from sklearn.model_selection import StratifiedShuffleSplit
import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from data.mod_zinc import ModZINC
from data.smnist import SMNIST
from utils import knn_filter, rwr_filter
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader


class FilterMaxNodes(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def __call__(self, data):
        return data.num_nodes <= self.max_nodes


class FilterConstant(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, data):
        data.x = torch.ones(data.num_nodes, self.dim)
        return data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_data(name, batch_size, rwr=False, cleaned=False):
    if name == 'ogbg-molhiv':
        data_train, data_val, data_test, max_num_nodes = get_molhiv()
        num_classes = 2
    elif name == 'ZINC':
        data_train, data_val, data_test = get_mod_zinc(rwr)
        max_num_nodes = 37
        num_classes = 1
    elif name == 'SMNIST':
        data_train, data_val, data_test = get_smnist(rwr)
        max_num_nodes = 75
        num_classes = 10
    else:
        data = get_tudataset(name, rwr, cleaned=cleaned)
        num_classes = data.num_classes
        max_num_nodes = 0
        for d in data:
            max_num_nodes = max(d.num_nodes, max_num_nodes)
        data_train, data_val, data_test = data_split(data)

    stats = dict()
    stats['num_features'] = data_train.num_node_features
    stats['num_classes'] = num_classes
    stats['max_num_nodes'] = max_num_nodes

    evaluator, encode_edge = (Evaluator(name), True) if name == 'ogbg-molhiv' else (None, False)

    train_loader = DataLoader(data_train, batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, stats, evaluator, encode_edge


def get_molhiv():
    path = osp.dirname(osp.realpath(__file__))
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=path)
    split_idx = dataset.get_idx_split()
    max_num_nodes = torch.tensor(dataset.data.num_nodes).max().item()
    return dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]], max_num_nodes


def get_tudataset(name, rwr, cleaned=False):
    transform = None
    if rwr:
        transform = rwr_filter
    path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''))
    dataset = TUDataset(path, name, pre_transform=transform, use_edge_attr=rwr, cleaned=cleaned)

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        dataset.transform = FilterConstant(10)#T.OneHotDegree(max_degree)    
    return dataset


def data_split(dataset):
    skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.data.y))[0]
    train_data = dataset[torch.from_numpy(train_idx)]

    skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

    val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.data.y[val_test_idx]))[0]
    val_data = dataset[torch.from_numpy(val_test_idx[val_idx])]
    test_data = dataset[torch.from_numpy(val_test_idx[test_idx])]
    return train_data, val_data, test_data


def get_smnist(rwr=True):
    name = 'SMNIST'
    if rwr:
        transforms = T.Compose([knn_filter, rwr_filter])
    else:
        transforms = knn_filter
    path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''), name)
    train = SMNIST(path, train=True, pre_transform=transforms) #, transform=T.ToDense(75))
    test = SMNIST(path, train=False, pre_transform=transforms) #, transform=T.ToDense(75))
    idxs = torch.randperm(len(train))
    val = train[idxs][:len(train) // 10]
    train = train[len(train) // 10:]
    return train, val, test


def get_mod_zinc(rwr):
    transform = None
    name = 'ModZINC'
    if rwr:
        transform = rwr_filter

    path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''), name)
    train = ModZINC(path, split='train', pre_transform=transform)
    test = ModZINC(path, split='test', pre_transform=transform)
    val = ModZINC(path, split='val', pre_transform=transform)
    return train, val, test
