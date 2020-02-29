# -*- coding: utf-8 -*-

'''
Visualize the updated embeddings of the graphsage module.
'''

import os
import numpy as np
from tsne import tsne
import matplotlib
matplotlib.use('Agg')   # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

import sys
sys.path.append('..')
from utils.data_utils import ZSL_Dataset
from utils.graph_utils import load_graph, adjMatrix2adjLists
from models.graphsage_visual import MeanAggregator, Encoder, SupervisedGraphSage
from models.gan import _netG, _netG2, _netD
from main import get_fake_unseen_visual_feat


def visualization(feature, label, save_dir, nameStr):
    '''t-SNE visualization for visual features.'''
    assert feature.shape[0] == label.shape[0]
    X = feature
    labels = label
    Y = tsne(X, 2, 50, 20.0)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    save_path = os.path.join(save_dir, nameStr + '.png')
    plt.savefig(save_path)
    print('visualization results are saved done in %s!' % save_dir)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = '../data'
    dataset_name = 'apy'
    mode = 'train'
    all_visualFea_label_file = 'res101.mat'
    auxiliary_file = 'original_att_splits.mat'
    use_pca = 'false'
    reduced_dim_pca = None

    zsl_dataset = ZSL_Dataset(root_dir, dataset_name, mode, all_visualFea_label_file, auxiliary_file, use_pca, reduced_dim_pca)
    te_data_unseen, te_data_seen = zsl_dataset.get_testData()
    te_vis_fea_unseen, te_sem_fea_unseen, te_label_unseen, te_labelID_unseen, te_sem_fea_pro_unseen = te_data_unseen
    te_vis_fea_seen, te_sem_fea_seen, te_label_seen, te_labelID_seen, te_sem_fea_pro_seen = te_data_seen
    tr_vis_fea, tr_sem_fea, all_tr_label, tr_labelID, tr_sem_fea_pro = zsl_dataset.get_trainData()

    save_rootdir = './visualization_output'
    save_dir = os.path.join(save_rootdir, dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('visualize {} dataset...'.format(dataset_name))
    ckpt_dir  = '../checkpoints'
    if dataset_name == 'awa':
        ckpt_file = 'checkpoint_awa_iterxxx.pkl'    # Customization
    elif dataset_name == 'apy':
        ckpt_file = 'checkpoint_apy_iterxxx.pkl'    # Customization
    else:
        pass
    ckpt_path = os.path.join(ckpt_dir, dataset_name, ckpt_file)
    ckpt_dict = torch.load(ckpt_path)
    graphsage_state_dict = ckpt_dict['state_dict']
    acc_unseen = ckpt_dict['acc_unseen']
    acc_seen = ckpt_dict['acc_seen']

    vi_fea_dim = zsl_dataset.vis_fea_dim
    se_fea_dim = zsl_dataset.sem_fea_dim

    graph_datadir='../data/preprocessed_data'
    adj_matrix = load_graph(dataset_name, graph_datadir)
    if dataset_name == 'awa':
        weight_threshold = 0.2
    elif dataset_name == 'apy':
        weight_threshold = 0.06
    adj_lists = adjMatrix2adjLists(adj_matrix, weight_threshold)
    all_labels = zsl_dataset.all_labels

    instanceIdx2classIdx = dict()
    classIdx2instanceIdx = defaultdict(set)
    for instanceIdx, classIdx in enumerate(all_labels):
        instanceIdx2classIdx[instanceIdx] = classIdx
        classIdx2instanceIdx[classIdx].add(instanceIdx)

    instanceIdx2classIdx_zsl_train = dict()
    classIdx2instanceIdx_zsl_train = defaultdict(set)
    for instanceIdx, classIdx in enumerate(all_tr_label):
        instanceIdx2classIdx_zsl_train[instanceIdx] = classIdx
        classIdx2instanceIdx_zsl_train[classIdx].add(instanceIdx)

    instanceIdx2classIdx_zsl_test_seen = dict()
    classIdx2instanceIdx_zsl_test_seen = defaultdict(set)
    for instanceIdx, classIdx in enumerate(te_label_seen):
        instanceIdx2classIdx_zsl_test_seen[instanceIdx] = classIdx
        classIdx2instanceIdx_zsl_test_seen[classIdx].add(instanceIdx)

    instanceIdx2classIdx_zsl_test_unseen = dict()
    classIdx2instanceIdx_zsl_test_unseen = defaultdict(set)
    for instanceIdx, classIdx in enumerate(te_label_unseen):
        instanceIdx2classIdx_zsl_test_unseen[instanceIdx] = classIdx
        classIdx2instanceIdx_zsl_test_unseen[classIdx].add(instanceIdx)

    use_z = 'true'
    z_dim = 100
    if use_z == 'true':
        netG = _netG(se_fea_dim, vi_fea_dim, z_dim).to(device)
    else:
        netG = _netG2(se_fea_dim, vi_fea_dim).to(device)
    gan_checkpoint_dir = '../gan_checkpoints'
    if dataset_name == 'awa':
        gan_checkpoint_name = 'checkpoint_awa_iter8401accUnseen57.34_accSeen72.49.pkl'
    elif dataset_name == 'apy':
        gan_checkpoint_name = 'checkpoint_apy_iter951_accUnseen29.80_accSeen64.53.pkl'
    else:
        pass
    sem_fea_pro = zsl_dataset.all_prototype_semantic_feature
    unseen_classes = te_labelID_unseen
    n_gene_perC = 40
    generated_vis_fea_dict = get_fake_unseen_visual_feat(netG, dataset_name, gan_checkpoint_dir, gan_checkpoint_name,
                                                         use_z, z_dim, sem_fea_pro, unseen_classes, n_gene_perC=n_gene_perC)


    firstHop_featureFunc = zsl_dataset.get_firstHop_featureFunc_visual_zsl_train()
    agg1 = MeanAggregator(firstHop_featureFunc).to(device)
    enc1 = Encoder(firstHop_featureFunc, vi_fea_dim, 128, adj_lists, agg1,
                   instanceIdx2classIdx_zsl_train, classIdx2instanceIdx_zsl_train,
                   instanceIdx2classIdx_zsl_test_seen, classIdx2instanceIdx_zsl_test_seen,
                   instanceIdx2classIdx_zsl_test_unseen, classIdx2instanceIdx_zsl_test_unseen,
                   classIdx2instanceIdx,
                   generated_vis_fea_dict,
                   mode='train', seen_labelID_set=tr_labelID,
                   gcn_style=True).to(device)
    enc1.num_samples = 10

    enc1.mode = 'test_unseen'

    n_classes = zsl_dataset.n_classes
    graphsage = SupervisedGraphSage(n_classes, enc1).to(device)

    graphsage.load_state_dict(graphsage_state_dict)
    graphsage = graphsage.to(device)
    graphsage.eval()

    te_unseen_indices = np.array(list(range(len(te_label_unseen))))
    _, embeddings = graphsage.forward(te_unseen_indices)
    embeddings = embeddings.cpu()
    embeddings = embeddings.detach().numpy().T
    print(np.isnan(embeddings).any())   # False
    nameStr_tgg = 'unseen_' + dataset_name + '_' + all_visualFea_label_file.split('.')[0] + '_tgg_nodeEmbedding'
    visualization(embeddings, te_label_unseen, save_dir, nameStr_tgg)



