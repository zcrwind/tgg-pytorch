# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle
from scipy import io as sio
import pickle
from sklearn.decomposition import PCA

from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data


'''
Use datasets following CVPR 2018 paper:
"Learning to compare: Relation network for few-shot learning".

see tgg-pytorch/utils/data_utils.py for more details.
'''

class ZSL_Dataset(Dataset):
    def __init__(self, root_dir, dataset_name, mode, all_visualFea_label_file, auxiliary_file, use_pca, reduced_dim_pca):
        super(ZSL_Dataset, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = dataset_name

        if dataset_name.lower() == 'cub':
            dataset_subdir = 'CUB1_data'
        elif dataset_name.lower() == 'awa':
            dataset_subdir = 'AwA1_data'
        elif dataset_name.lower() == 'apy':
            dataset_subdir = 'APY_data'
        elif dataset_name.lower() == 'sun':
            dataset_subdir = 'SUN_data'
        else:
            raise RuntimeError('Unknown dataset of %s' % dataset_name)

        # visual feature for whole dataset
        all_visFea_label_path = os.path.join(root_dir, dataset_subdir, all_visualFea_label_file)
        all_visFea_label_data = sio.loadmat(all_visFea_label_path)
        all_vis_fea = all_visFea_label_data['features'].astype(np.float32).T

        ## prepare label
        all_labels = all_visFea_label_data['labels'].astype(np.long).squeeze() - 1   # from 1-based to 0-based

        auxiliary_path = os.path.join(root_dir, dataset_subdir, auxiliary_file)
        auxiliary_data = sio.loadmat(auxiliary_path)
        tr_vis_fea_idx = auxiliary_data['trainval_loc'].squeeze() - 1
        te_vis_fea_idx_seen = auxiliary_data['test_seen_loc'].squeeze() - 1
        te_vis_fea_idx_unseen = auxiliary_data['test_unseen_loc'].squeeze() - 1
        all_prototype_semantic_feature = auxiliary_data['att'].astype(np.float32).T

        ## prepare training data
        self.tr_vis_fea = all_vis_fea[tr_vis_fea_idx]
        self.tr_label = all_labels[tr_vis_fea_idx]
        self.tr_sem_fea = all_prototype_semantic_feature[self.tr_label]
        self.tr_labelID = np.unique(self.tr_label)
        self.n_tr_class = self.tr_labelID.shape[0]
        self.tr_sem_fea_pro = all_prototype_semantic_feature[self.tr_labelID]
        assert self.tr_vis_fea.shape[0] == self.tr_sem_fea.shape[0] == self.tr_label.shape[0]

        ## prepare test data ##
        # unseen
        self.te_vis_fea_unseen = all_vis_fea[te_vis_fea_idx_unseen]
        self.te_label_unseen = all_labels[te_vis_fea_idx_unseen]
        self.te_sem_fea_unseen = all_prototype_semantic_feature[self.te_label_unseen]
        self.te_labelID_unseen = np.unique(self.te_label_unseen)
        self.te_sem_fea_pro_unseen = all_prototype_semantic_feature[self.te_labelID_unseen]
        
        assert self.te_vis_fea_unseen.shape[0] == self.te_label_unseen.shape[0] == self.te_sem_fea_unseen.shape[0]
        assert self.te_labelID_unseen.shape[0] == self.te_sem_fea_pro_unseen.shape[0]
        # seen
        self.te_vis_fea_seen = all_vis_fea[te_vis_fea_idx_seen]
        self.te_label_seen = all_labels[te_vis_fea_idx_seen]
        self.te_sem_fea_seen = all_prototype_semantic_feature[self.te_label_seen]
        self.te_labelID_seen = np.unique(self.te_label_seen)
        self.te_sem_fea_pro_seen = all_prototype_semantic_feature[self.te_labelID_seen]
        assert self.te_vis_fea_seen.shape[0] == self.te_label_seen.shape[0] == self.te_sem_fea_seen.shape[0]
        assert self.te_labelID_seen.shape[0] == self.te_sem_fea_pro_seen.shape[0]


        tr_vis_fea = self.tr_vis_fea
        te_vis_fea_unseen = self.te_vis_fea_unseen
        te_vis_fea_seen = self.te_vis_fea_seen
        if use_pca == 'true':
            n_components = reduced_dim_pca
            if n_components < tr_vis_fea.shape[1]:
                raise(RuntimeError('visual feature dim < the dim that PCA will reduced to!'))
            tr_vis_fea = pca(n_components, tr_vis_fea)
            te_vis_fea_unseen = pca(n_components, te_vis_fea_unseen)
            te_vis_fea_seen = pca(n_components, te_vis_fea_seen)
        tr_vis_fea = (tr_vis_fea - tr_vis_fea.mean()) / tr_vis_fea.var()
        te_vis_fea_unseen = (te_vis_fea_unseen - te_vis_fea_unseen.mean()) / te_vis_fea_unseen.var()
        te_vis_fea_seen = (te_vis_fea_seen - te_vis_fea_seen.mean()) / te_vis_fea_seen.var()
        self.tr_vis_fea = tr_vis_fea
        self.te_vis_fea_unseen = te_vis_fea_unseen
        self.te_vis_fea_seen = te_vis_fea_seen

        self.all_prototype_semantic_feature = all_prototype_semantic_feature
        self.vis_fea_dim = self.tr_vis_fea.shape[1]
        self.sem_fea_dim = self.all_prototype_semantic_feature.shape[1]
        self.tr_classIdx_map = dict(zip(self.tr_labelID, range(len(self.tr_labelID))))
        self.te_classIdx_unseen_map = dict(zip(self.te_labelID_unseen, range(len(self.te_labelID_unseen))))
        self.te_classIdx_seen_map = dict(zip(self.te_labelID_seen, range(len(self.te_labelID_seen))))
 

    def __getitem__(self, index):
        if self.mode == 'train':
            vis_fea = self.tr_vis_fea[index]
            sem_fea = self.tr_sem_fea[index]
            label = self.tr_classIdx_map[self.tr_label[index]]
            return vis_fea, sem_fea, label
        elif self.mode == 'test':
            vis_fea_unseen = self.te_vis_fea_unseen[index]
            sem_fea_unseen = self.te_sem_fea_unseen[index]
            label_unseen = self.te_classIdx_unseen_map[self.te_label_unseen[index]]
            vis_fea_seen = self.te_vis_fea_seen[index]
            sem_fea_seen = self.te_sem_fea_seen[index]
            label_seen = self.te_classIdx_seen_map[self.te_label_seen[index]]
            return vis_fea_unseen, sem_fea_unseen, label_unseen, vis_fea_seen, sem_fea_seen, label_seen


    def __len__(self):
        if  self.mode == 'train':
            return self.tr_vis_fea.shape[0]
        elif self.mode == 'test':
            return (self.te_vis_fea_unseen.shape[0], self.te_vis_fea_seen.shape[0])


    def get_testData(self):
        te_unseen = (self.te_vis_fea_unseen, self.te_sem_fea_unseen, self.te_label_unseen, self.te_labelID_unseen, self.te_sem_fea_pro_unseen)
        te_seen   = (self.te_vis_fea_seen, self.te_sem_fea_seen, self.te_label_seen, self.te_labelID_seen, self.te_sem_fea_pro_seen)
        return te_unseen, te_seen

    def get_trainData(self):
        return self.tr_vis_fea, self.tr_sem_fea, self.tr_label, self.tr_labelID, self.tr_sem_fea_pro


    def get_tr_centroid(self):
        '''Get the centroid of each class in training set.'''
        tr_cls_centroid = np.zeros([self.n_tr_class, self.tr_vis_fea.shape[1]]).astype(np.float32)
        for i in range(self.n_tr_class):
            current_tr_classId = self.tr_labelID[i]
            tr_cls_centroid[i] = np.mean(self.tr_vis_fea[self.tr_label == current_tr_classId], axis=0)
        return tr_cls_centroid


def pca(n_components, data):
    '''PCA for visual feature dimension reduction.'''
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_new = pca.transform(data)
    print('data.shape after PCA:', data_new.shape)
    return data_new



if __name__ == '__main__':
    root_dir = '/home/zhangchenrui/workspace/mm2019/tgg-pytorch/data'
    all_visualFea_label_file = 'res101.mat'
    auxiliary_file = 'original_att_splits.mat'

    dataset_name = 'awa'
    mode = 'train'
    use_pca = 'false'
    reduced_dim_pca = None
    zsl_dataset = ZSL_Dataset(root_dir, dataset_name, mode, all_visualFea_label_file, auxiliary_file, use_pca, reduced_dim_pca)
    zsl_dataloader = data.DataLoader(zsl_dataset, batch_size=64, shuffle=False, num_workers=2)
    for step, (vis_fea, sem_fea, label) in enumerate(zsl_dataloader):
        print('vis_fea.shape, sem_fea.shape, label.shape', vis_fea.shape, sem_fea.shape, label.shape)
        break

