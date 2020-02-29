# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import random
from termcolor import cprint
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

'''
Module(s) for aggregating embeddings of neighbors.
Here only use the mean aggregator.
'''

class MeanAggregator(nn.Module):
    '''
    Aggregates a node's embeddings using mean of neighbors' embeddings
    '''
    def __init__(self, features_func, gcn_style=False): 
        '''
        Initializes the aggregator for a specific graph.
        Args:
            features_func:
                function mapping LongTensor of node ids to FloatTensor of feature values.
            gcn_style:
                whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style.
        '''
        super(MeanAggregator, self).__init__()
        self.features_func = features_func
        self.gcn_style = gcn_style
        
    def forward(self, nodes, to_neighs, classIdx2instanceIdx, \
                seen_labelID_set, embed_dim, generated_vis_fea_dict, \
                num_sample=10):
        '''
        Args:
            nodes:
                list of nodes in a batch
            to_neighs:
                list of sets, each set is the set of neighbors for node in batch
            num_sample:
                number of neighbors to sample. No sampling if None.
            embed_dim:
                the dimension of neighbors' embeddings. For agg1, it is the dim of the input visual features.
            generated_vis_fea_dict:
                a dict that contains synthesized visual features of unseen classes
                whose keys are global indexes of unseen classes.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn_style:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        assert len(unique_nodes_list) > 0

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        
        mask[row_indices, column_indices] = 1
        mask = mask.to(device)
        num_neigh = mask.sum(1, keepdim=True)

        embed_matrix = np.zeros((len(unique_nodes_list), embed_dim))
        for i, classIdx in enumerate(unique_nodes_list):
            if classIdx in seen_labelID_set:
                instanceIdx_set = classIdx2instanceIdx[classIdx]
                n_instance = len(instanceIdx_set)
                assert n_instance > 0
                instanceIdx_list = list(instanceIdx_set)
                which = random.randint(0, n_instance - 1)
                selected_instanceIdx = instanceIdx_list[which]
                feat = self.features_func(torch.LongTensor([selected_instanceIdx]).to(device))
                embed_matrix[i] = feat

            else:   # unseen classes --> load generated fake visual feature
                generated_vis_fea_of_currentClass = generated_vis_fea_dict[classIdx]
                assert isinstance(generated_vis_fea_of_currentClass, list)
                assert len(generated_vis_fea_of_currentClass) > 0
                selected_instanceIdx = random.randint(0, len(generated_vis_fea_of_currentClass) - 1)
                fake_vis_fea = generated_vis_fea_of_currentClass[selected_instanceIdx]
                assert fake_vis_fea.shape[-1] == embed_dim
                embed_matrix[i] = fake_vis_fea

        if str(type(embed_matrix)) == "<class 'numpy.ndarray'>":
            embed_matrix = torch.from_numpy(embed_matrix)
        
        embed_matrix = embed_matrix.to(device)
        embed_matrix = embed_matrix.float()
        to_feats = mask.mm(embed_matrix)
        return to_feats


class Encoder(nn.Module):
    '''
        Encodes a node using 'convolutional' GraphSage approach.
    '''
    def __init__(self, features_func, feature_dim, embed_dim, adj_lists, aggregator,
                 instanceIdx2classIdx_zsl_train, classIdx2instanceIdx_zsl_train,
                 instanceIdx2classIdx_zsl_test_seen, classIdx2instanceIdx_zsl_test_seen,
                 instanceIdx2classIdx_zsl_test_unseen, classIdx2instanceIdx_zsl_test_unseen,
                 classIdx2instanceIdx, generated_vis_fea_dict,
                 mode='train', seen_labelID_set=None,
                 num_sample=10, base_model=None, gcn_style=False, feature_transform=False):
        '''
        Args:
            feature_dim:
                The dim of Encoder's input.
            embed_dim:
                The dim of Encoder's output.
            mode:
                `train` or `test_seen` or `test_unseen`
        '''
        super(Encoder, self).__init__()

        self.features_func = features_func
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn_style = gcn_style
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim if self.gcn_style else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)
        self.bn = nn.BatchNorm1d(self.embed_dim)

        self.classIdx2instanceIdx = classIdx2instanceIdx
        self.instanceIdx2classIdx_zsl_train = instanceIdx2classIdx_zsl_train
        self.classIdx2instanceIdx_zsl_train = classIdx2instanceIdx_zsl_train
        self.instanceIdx2classIdx_zsl_test_seen = instanceIdx2classIdx_zsl_test_seen
        self.classIdx2instanceIdx_zsl_test_seen = classIdx2instanceIdx_zsl_test_seen
        self.instanceIdx2classIdx_zsl_test_unseen = instanceIdx2classIdx_zsl_test_unseen
        self.classIdx2instanceIdx_zsl_test_unseen = classIdx2instanceIdx_zsl_test_unseen
        assert mode in ['train', 'test_seen', 'test_unseen']
        self.mode = mode
        self.seen_labelID_set = seen_labelID_set
        self.generated_vis_fea_dict = generated_vis_fea_dict


    def forward(self, nodes):
        '''
        Generates embeddings for a batch of nodes.
        Args:
            nodes:
                list of nodes
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy()
        if self.mode == 'train':
            nodes = [self.instanceIdx2classIdx_zsl_train[instanceIdx] for instanceIdx in nodes]
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                                  self.classIdx2instanceIdx_zsl_train, self.seen_labelID_set,
                                                  self.feat_dim, self.generated_vis_fea_dict,
                                                  self.num_sample)
        elif self.mode == 'test_seen':
            print('It is test_seen mode!!', end='    ')
            nodes = [self.instanceIdx2classIdx_zsl_test_seen[instanceIdx] for instanceIdx in nodes]
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                                  self.classIdx2instanceIdx_zsl_test_seen, self.seen_labelID_set,
                                                  self.feat_dim, self.generated_vis_fea_dict,
                                                  self.num_sample)
        else:
            print('It is test_unseen mode!!', end='  ')
            nodes = [self.instanceIdx2classIdx_zsl_test_unseen[instanceIdx] for instanceIdx in nodes]
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                                  self.classIdx2instanceIdx_zsl_train, self.seen_labelID_set,
                                                  self.feat_dim, self.generated_vis_fea_dict,
                                                  self.num_sample)

        if not self.gcn_style:
            self_feats = self.features_func(torch.LongTensor(nodes).to(device))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.bn(self.weight.mm(combined.t()).t()).t())    # batch normalization is crucial for the AwA dataset.
        
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, encoder, LP_input_dim):
        super(SupervisedGraphSage, self).__init__()
        self.encoder = encoder
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, encoder.embed_dim))
        init.xavier_uniform_(self.weight)

        self.LP_mlp = nn.Sequential(
            nn.Linear(LP_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def cal_sigma(self, embeds):
        '''the size of embeds: [batchsize x embed_dim]'''
        sigma = self.LP_mlp(embeds)
        return sigma

    def forward(self, nodes):
        embeds = self.encoder(nodes)
        scores = self.weight.mm(embeds)
        sigma = self.cal_sigma(torch.t(embeds))
        return scores.t(), embeds, sigma

    def loss(self, nodes, labels):
        scores, _, _ = self.forward(nodes)
        return self.xent(scores, labels.squeeze())