# -*- coding: utf-8 -*-

'''
common tools
'''

import os
import pickle
import numpy as np
import scipy.io as sio
import random
import torch
from torch.nn import init


def pklLoad(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def pklSave(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

def nets_weights_init(nets):
    for net in nets:
        net.apply(weights_init)

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def print_nets(nets):
    for net in nets:
        print(net)

def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def print_args(args):
    print('-' * 50)
    for arg, content in args.__dict__.items():
        print("{}: {}".format(arg, content))
    print('-' * 50)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = True   # Speed up training when the training set changes little