from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pickle


# preprocessing
def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
    
def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value>threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index], 
        (edge_tmp[preserved_edge_index,0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network
    

class DiseaseGeneDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DiseaseGeneDataset, self).__init__(root, transform,
                                                 pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        data_list = []

        data = Data(x=feature_matrix, edge_index=edge_index, y=label)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices),self.processed_paths[0])
    
