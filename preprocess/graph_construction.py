# -*- coding: utf-8 -*-

'''
Construct the knowledge graph (adjacency matrix) of classes in different datasets.
Run only once.
'''

import os
import re
import numpy as np
import pickle
import requests
from scipy import io as sio
from collections import defaultdict
import sys
sys.path.append('..')
from utils.tools import pklSave

'''
# <tiny checking before graph construction>

class1 = 'cat'
class2 = 'dog'

relation_search_request = 'http://api.conceptnet.io/related/c/en/'
request_url = relation_search_request + class1 + '?filter=/c/en/' + class2

flag = True
while flag:
    try:
        obj = requests.get(request_url).json()
    except:
        print('There are some errors in the retrieving process! class1=%s, class2=%s' % (class1, class2))
    else:
        flag = False

print('obj:', obj)  # {'@id': '/c/en/cat', 'related': [{'@id': '/c/en/dog', 'weight': 0.579}]}

if 'related' in obj:
    if len(obj['related']):
        # adj_matrix[i, j] += obj['related'][0]['weight']
        print('class1=%s, class2=%s, weight=%f' % (class1, class2, obj['related'][0]['weight']))
'''

''' The template of crawler
for i in range(num_nodes_action):
    for j in range(num_nodes_object):
        action = action_entities[i]
        object_tmp = object_entities[j]
        for ae in action:
            request_tmp = relation_search_request + ae + '?filter=/c/en/' + object_tmp
            flag = 1
            while flag:
                try:
                    obj = requests.get(request_tmp).json()
                except:
                    print "There are some errors in the retrieving process! action=%s, object=%s"% (action, object_tmp)
                else:
                    flag = 0

            if obj.has_key('related'):
                if len(obj['related']):
                    adj_matrix[i,j] += obj['related'][0]['weight']
                    print "action=%s, object=%s, weight=%f" % (action, object_tmp, obj['related'][0]['weight'])
'''

def check_adj(adj_matrix, idx2classname, doprint=True):
    '''
        check the adj matrix for 0 items (where the weight of edge is 0)
    '''
    zero_indexes = np.where(adj_matrix == 0.)
    row_index, col_index = zero_indexes
    assert len(row_index) == len(col_index)
    n_zeros_directed = len(row_index)
    print('There are {} zero edges in adj matrix.'.format(n_zeros_directed))
    zero_edges = []
    for row, col in zip(row_index, col_index):
        _edge = (row, col)
        edge_ = (col, row)
        if _edge in zero_edges or edge_ in zero_edges:
            continue
        else:
            zero_edges.append(_edge)
    print('There are {} undirected zero edges in adj matrix:'.format(len(zero_edges)))

    if doprint:
        for edge in zero_edges:
            row, col = edge
            class1 = idx2classname[row]
            class2 = idx2classname[col]
            print('{} <--> {}'.format(class1, class2))



if __name__ == '__main__':
    root_datadir = '../data'
    dataset_name = 'apy'    # 'cub' / 'awa' / 'apy' / 'sun'
    assert dataset_name in ['cub', 'awa', 'apy', 'sun']

    if dataset_name == 'cub':
        dataset_subdir = 'CUB1_data'
    elif dataset_name == 'awa':
        dataset_subdir = 'AwA1_data'
    elif dataset_name == 'apy':
        dataset_subdir = 'APY_data'
    elif dataset_name == 'sun':
        dataset_subdir = 'SUN_data'
    else:
        raise RuntimeError('Unknown dataset of %s' % dataset_name)

    ## for AwA or CUB
    if dataset_name in ['cub', 'awa']:
        auxiliary_file = 'original_att_splits.mat'  # `auxiliary_file`: contains split file and semantic feature.
        auxiliary_path = os.path.join(root_datadir, dataset_subdir, auxiliary_file)
        auxiliary_data = sio.loadmat(auxiliary_path)
        allclasses_names = auxiliary_data['allclasses_names']
        allclasses_nameStr = [allclasses_names[i][0][0] for i in range(len(allclasses_names))]
        print(allclasses_nameStr)
        '''
        e.g., awa:
        ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse', 'german+shepherd',
         'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', 'spider+monkey',
         'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel',
         'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra',
         'giant+panda', 'deer', 'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow',
         'dolphin']
        '''
        n_classes = len(allclasses_names)
        adj_matrix = np.zeros((n_classes, n_classes))
        classname2idx = dict()
        idx2classname = dict()
        for i in range(n_classes):
            classname = allclasses_nameStr[i]
            classname2idx[classname] = i
            idx2classname[i] = classname

        relation_search_request = 'http://api.conceptnet.io/related/c/en/'
        for row in range(n_classes):
            for col in range(row, n_classes):  # Symmetric matrix
                class1 = allclasses_nameStr[row]
                class2 = allclasses_nameStr[col]
                request_url = relation_search_request + class1 + '?filter=/c/en/' + class2
                flag = True
                while flag:
                    try:
                        obj = requests.get(request_url).json()
                    except:
                        print('There are some errors in the retrieving process! class1=%s, class2=%s' % (class1, class2))
                    else:
                        flag = False

                if 'related' in obj:
                    if len(obj['related']):
                        adj_matrix[row, col] += obj['related'][0]['weight']
                        print('class1=%s, class2=%s, weight=%f' % (class1, class2, obj['related'][0]['weight']))

        for row in range(1, n_classes):
            for col in range(0, row - 1):
                adj_matrix[row][col] = adj_matrix[col][row]
        print('adj_matrix', adj_matrix)
        check_adj(adj_matrix, idx2classname, doprint=False)
        save_dir = os.path.join(root_datadir, 'preprocessed_data', dataset_name.lower())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        adj_filename = dataset_name + '_class_adj_' + 'byConceptNet5.5_.pkl'
        adj_filepath = os.path.join(save_dir, adj_filename)
        pklSave(adj_filepath, adj_matrix)
        print('save {} done!'.format(adj_filename))


    ## for ImageNet
    elif dataset_name == 'imagenet':
        print('deal with ImageNet dataset...')
        auxiliary_dir = os.path.join(root_datadir, dataset_subdir, 'extra_files_byZCR')
        all_words_file = 'words.txt'

        id2classname = defaultdict(set) # NOTE: `id` here is something like `n02815834`
        words_path = os.path.join(auxiliary_dir, all_words_file)
        with open(words_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            fragments = re.split(r'[\t,]', line.strip())
            _id = fragments[0]
            current_classes = [item.strip() for item in fragments[1:]]
            id2classname[_id] = set(current_classes)

        id2classname_to_use = defaultdict(set)
        dir1000 = os.path.join(root_datadir, dataset_subdir, '1000_360')
        dir800  = os.path.join(root_datadir, dataset_subdir, '800_200')

        classes_file_800 = os.path.join(dir800, 'imnet2010_800.txt')
        classes_file_200 = os.path.join(dir800, 'imnet2010_200.txt')

        with open(classes_file_800, 'r') as f:
            _lines = f.readlines()
        id_list_800 = [_id.strip() for _id in _lines]

        with open(classes_file_200, 'r') as f:
            _lines = f.readlines()
        id_list_200 = [_id.strip() for _id in _lines]

        all_ids = id_list_800 + id_list_200
        for _id in all_ids:
            id2classname_to_use[_id] = id2classname[_id]

        n_classes = len(id2classname_to_use)
        adj_matrix = np.zeros((n_classes, n_classes))

        relation_search_request = 'http://api.conceptnet.io/related/c/en/'
        for row in range(n_classes - 1):
            for col in range((row + 1), n_classes):  # Symmetric matrix
                class_id1  = all_ids[row]
                class_id2  = all_ids[col]
                class_set1 = id2classname_to_use[class_id1]
                class_set2 = id2classname_to_use[class_id2]

                ## get the score between two sets: class_set1 and class_set2
                local_cnt = 0   # for normalization
                for class1 in class_set1:
                    for class2 in class_set2:
                        request_url = relation_search_request + class1 + '?filter=/c/en/' + class2
                        flag = True
                        while flag:
                            try:
                                obj = requests.get(request_url).json()
                            except:
                                print('There are some errors in the retrieving process! class1=%s, class2=%s' % (class1, class2))
                            else:
                                flag = False

                        if 'related' in obj:
                            if len(obj['related']):
                                adj_matrix[row, col] += obj['related'][0]['weight']
                                print('class1=%s, class2=%s, weight=%f' % (class1, class2, obj['related'][0]['weight']))
                                local_cnt += 1

                if local_cnt > 0:
                    adj_matrix[row, col] /= local_cnt
                    local_cnt = 0

        for row in range(1, n_classes):
            for col in range(0, row - 1):
                adj_matrix[row][col] = adj_matrix[col][row]
        print('adj_matrix', adj_matrix)
        check_adj(adj_matrix)
        save_dir = os.path.join(root_datadir, 'preprocessed_data', dataset_name.lower())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        adj_filename = dataset_name + '_class_adj_' + 'byConceptNet5.5_.pkl'
        adj_filepath = os.path.join(save_dir, adj_filename)
        pklSave(adj_filepath, adj_matrix)
        print('save {} done!'.format(adj_filename))


    ## for APY dataset
    elif dataset_name == 'apy':
        tr_classes_file = 'trainvalclasses.txt'
        tr_classes_path = os.path.join(root_datadir, dataset_subdir, tr_classes_file)
        with open(tr_classes_path, 'r') as f:
            lines = f.readlines()
        tr_classes = [line.strip() for line in lines]

        te_classes_file = 'testclasses.txt'
        te_classes_path = os.path.join(root_datadir, dataset_subdir, te_classes_file)
        with open(te_classes_path, 'r') as f:
            lines = f.readlines()
        te_classes = [line.strip() for line in lines]

        allclasses_nameStr = tr_classes + te_classes
        # print('allclasses_nameStr', allclasses_nameStr)
        '''
        ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'dog',
         'monkey', 'wolf', 'zebra', 'mug', 'building', 'bag', 'carriage', 'sofa', 'centaur',
         'diningtable', 'tvmonitor', 'goat', 'motorbike', 'cow', 'jetski', 'train', 'sheep',
         'statue', 'horse', 'person', 'pottedplant', 'donkey']
        '''

        n_classes = len(allclasses_nameStr)
        adj_matrix = np.zeros((n_classes, n_classes))

        classname2idx = dict()
        idx2classname = dict()
        for i in range(n_classes):
            classname = allclasses_nameStr[i]
            classname2idx[classname] = i
            idx2classname[i] = classname


        relation_search_request = 'http://api.conceptnet.io/related/c/en/'
        start_time = time.time()
        for row in range(n_classes):
            for col in range(row, n_classes):  # Symmetric matrix
                class1 = allclasses_nameStr[row]
                class2 = allclasses_nameStr[col]
                request_url = relation_search_request + class1 + '?filter=/c/en/' + class2
                flag = True
                while flag:
                    try:
                        obj = requests.get(request_url).json()
                    except:
                        print('There are some errors in the retrieving process! class1=%s, class2=%s' % (class1, class2))
                    else:
                        flag = False

                if 'related' in obj:
                    if len(obj['related']):
                        adj_matrix[row, col] += obj['related'][0]['weight']
                        print('class1=%s, class2=%s, weight=%f' % (class1, class2, obj['related'][0]['weight']))

        print('times used: {}'.format(time.time() - start_time))
        for row in range(1, n_classes):
            for col in range(0, row - 1):
                adj_matrix[row][col] = adj_matrix[col][row]
        print('adj_matrix', adj_matrix)
        check_adj(adj_matrix, idx2classname)
        save_dir = os.path.join(root_datadir, 'preprocessed_data', dataset_name.lower())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        adj_filename = dataset_name + '_class_adj_' + 'byConceptNet5.5_.pkl'
        adj_filepath = os.path.join(save_dir, adj_filename)
        pklSave(adj_filepath, adj_matrix)
        print('save {} done!'.format(adj_filename))

