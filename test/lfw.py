#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

import pdb

total = 7701
list_file = "data/lfw/data/verification.txt"
root = "data/lfw/data/tar/align"

__all__ = ["test_lfw"]

def get_pairs(path_):
    '''read pairs from file'''
    with open(path_) as fin:
        pairs = [list(map(int, l.strip().split(' '))) for l in fin.readlines()]
    pairs = np.array(pairs)
    pairs[:, 0:2] -= 1
    return pairs

def distance(disa, disb):
    from scipy.spatial.distance import cosine
    assert disa.shape == disa.shape
    return np.array([cosine(disa[i, :], disb[i, :]) for i in range(disa.shape[0])])

def tune_accuracy(feat, pairs, mask):
    pairs_train = pairs[mask, :]
    pairs_val = pairs[~mask, :]
    dist_train = distance(feat[pairs_train[:, 0], :], feat[pairs_train[:, 1], :])
    dist_val = distance(feat[pairs_val[:, 0], :], feat[pairs_val[:, 1], :])
    acc_best = 0
    th_best = -1
    for th in np.arange(0, 2, 0.001):
        acc_train = ((dist_train < th) == pairs[mask, 2]).mean()
        if acc_train > acc_best:
            th_best  = th
            acc_best = acc_train
    return ((dist_val < th_best) == pairs[~mask, 2]).mean()

def verification(feat, pairs, fold = 10):
    acc = []
    n = pairs.shape[0]
    fold_size = int(n / fold);
    for i in range(fold):
        mask = np.ones((n))
        mask[i*fold_size : (i+1)*fold_size] = 0
        acc.append(tune_accuracy(feat, pairs, mask.astype(np.bool)))
    acc = np.array(acc)
    return acc.mean(), acc.std()

def test_lfw(features):
    pairs = get_pairs('data/lfw/data/verification/id_pair_7701.txt')
    rst = verification(features, pairs)
    return rst

def build_testset():
    with open(list_file, 'r') as f:
        lines = f.readlines()
        fns = [os.path.join(root, l.strip()) for l in lines]
    return fns[:total]
