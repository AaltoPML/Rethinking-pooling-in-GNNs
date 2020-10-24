import os

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)


class ModZINC(InMemoryDataset):

    url = 'http://www.amauriholanda.org/modzinc.zip'

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(ModZINC, self).__init__(root, transform, pre_transform,
                                               pre_filter)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['training.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'val.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            x, node_slice, edge_slice, edge_attr, edge_index, y = torch.load(raw_path)

            new_x = torch.zeros(x.size(0), 28)
            new_x[torch.arange(x.size(0)), x.view(-1)] = 1.0
            x = new_x

            m = y.size(0)
            graph_slice = torch.arange(m + 1, dtype=torch.long)
            self.data = Data(x=x, y=y, edge_index=edge_index.T, edge_attr=edge_attr)
            self.slices = {
                'x': node_slice,
                'y': graph_slice,
                'edge_index': edge_slice,
                'edge_attr': edge_slice
            }

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)

            torch.save((self.data, self.slices), path)
