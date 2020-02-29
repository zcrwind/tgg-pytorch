# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import time
import pickle
import networkx as nx
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('.')
from .tools import pklLoad


def load_graph(dataset_name, graph_datadir, self_loop=False):
    '''
        Load the graph data (adj matrix with weighted edges) from .pkl file.
    '''
    suffix = '_class_adj_byConceptNet5.5_.pkl'
    grpah_file = dataset_name + suffix
    graph_path = os.path.join(graph_datadir, dataset_name, grpah_file)
    graph_data = pklLoad(graph_path)
    adj_matrix = graph_data

    N = adj_matrix.shape[0]
    for i in range(N):
        if self_loop:
            adj_matrix[i][i] = 1
        else:
            adj_matrix[i][i] = 0

    return adj_matrix


def adjMatrix2adjLists(adj_matrix, threshold):
    '''
        Transform adj matrix to adj lists (for graphsage)
    '''
    assert threshold <= 1
    sparse_adj_matrix = np.where(adj_matrix > threshold, adj_matrix, 0)
    adj_lists = defaultdict(set)

    nonzero_idx = np.argwhere(sparse_adj_matrix > 0)
    n_edges = nonzero_idx.shape[0]
    N = adj_matrix.shape[0]
    print('edges: {}'.format(n_edges))
    print('edges ratio: {}'.format(n_edges / (N * N)))
    assert len(sparse_adj_matrix.nonzero()[0]) == nonzero_idx.shape[0]

    for i in range(nonzero_idx.shape[0]):
        source_nodeidx = nonzero_idx[i][0]
        target_nodeidx = nonzero_idx[i][1]
        adj_lists[source_nodeidx].add(target_nodeidx)
        adj_lists[target_nodeidx].add(source_nodeidx)

    return adj_lists


def visualization(dataset_name, weight_threshold, adj_lists, save_dir):
    edges = []
    for k in adj_lists.keys():
        source_node = k
        target_node_list = adj_lists[k]
        print(len(target_node_list))
        for target_node in target_node_list:
            edges.append((source_node, target_node))

    graph = nx.Graph()
    graph.add_edges_from(edges)
    nx.draw(graph, with_labels=True, font_weight='bold')
    fig_name = dataset_name + '_thres' + str(weight_threshold) + '_graph.png'
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)



if __name__ == '__main__':
    dataset_name = 'apy'
    suffix = '_class_adj_byConceptNet5.5_.pkl'
    graph_datadir = '../data/preprocessed_data'
    adj_matrix = load_graph(dataset_name, graph_datadir)
    weight_threshold = 0.08
    adj_lists = adjMatrix2adjLists(adj_matrix, weight_threshold)
    save_dir = '../results'
    visualization(dataset_name, weight_threshold, adj_lists, save_dir)




