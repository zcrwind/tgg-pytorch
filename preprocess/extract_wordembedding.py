# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import time
import scipy.io as sio
sys.path.append('..')
from utils.tools import pklSave


'''
- APY_data
    original_att_splits.mat
    res101.mat
    testclasses.txt
    trainvalclasses.txt

- AwA1_data
    original_att_splits.mat
    res101.mat

- CUB1_data
    original_att_splits.mat
    res101.mat

- SUN_data
    original_att_splits.mat
    res101.mat
    testclasses.txt
    trainvalclasses.txt
'''

if __name__ == '__main__':
    raw_data_dir = './raw_data'
    root_datadir = '../data'
    datasets = ['awa', 'cub', 'apy', 'sun']
    subdirs  = ['AwA1_data', 'CUB1_data', 'APY_data', 'SUN_data']
    assert len(datasets) == len(subdirs)
    dataset_subdir_dict = {dataset : subdir for dataset, subdir in zip(datasets, subdirs)}

    classnames_all_datasets = []
    for dataset in datasets:
        dataset_subdir = dataset_subdir_dict[dataset]

        if dataset == 'awa' or dataset == 'cub':
            auxiliary_file = 'original_att_splits.mat'
            auxiliary_path = os.path.join(root_datadir, dataset_subdir, auxiliary_file)
            auxiliary_data = sio.loadmat(auxiliary_path)
            allclasses_names = auxiliary_data['allclasses_names']
            allclasses_nameStr = [allclasses_names[i][0][0] for i in range(len(allclasses_names))]
            if dataset == 'awa':
                for c in allclasses_nameStr:
                    li = c.split('+')
                    classnames_all_datasets += li
            if dataset == 'cub':
                for c in allclasses_nameStr:
                    li = c.split('.')
                    assert len(li) == 2
                    class_name = li[-1]
                    items = class_name.split('_')
                    classnames_all_datasets += items

            print(dataset)
            print(allclasses_nameStr)
            print('+' * 50)

        elif dataset == 'apy' or dataset == 'sun':
            test_class_file = 'testclasses.txt'
            test_class_path = os.path.join(root_datadir, dataset_subdir, test_class_file)
            with open(test_class_path, 'r') as f:
                lines = f.readlines()
            test_classes_nameStr = [line.strip() for line in lines]
            if dataset == 'apy':
                classnames_all_datasets += test_classes_nameStr
            else:
                for c in test_classes_nameStr:
                    li = c.split('_')
                    classnames_all_datasets += li
            print(dataset)
            print(test_classes_nameStr)
            print('+' * 50)

            trainval_class_file = 'trainvalclasses.txt'
            trainval_class_path = os.path.join(root_datadir, dataset_subdir, trainval_class_file)
            with open(trainval_class_path, 'r') as f:
                lines = f.readlines()
            trainval_class_nameStr = [line.strip() for line in lines]
            if dataset == 'apy':
                classnames_all_datasets += trainval_class_nameStr
            else:
                for c in trainval_class_nameStr:
                    li = c.split('_')
                    classnames_all_datasets += li
            print(dataset)
            print(trainval_class_nameStr)
            print('+' * 50)

    print(len(classnames_all_datasets))
    classnames_all_datasets = set(classnames_all_datasets)
    print('%' * 60)
    print(classnames_all_datasets)
    print(len(classnames_all_datasets))

