#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-all
import sys
import os, sys
import numpy as np

total = 14038
list_file = "data/ijba/labels/idxs/img_list.txt"
root = "data/ijba/align_image/joined/"

__all__ = ["test_ijba"]

def load_pair(split_file):
    with open(split_file) as fin:
        a = fin.read()
    pairs = []
    for pair in a.split('#')[1:]:
        pack = pair.split('\n')
        if len(pack) != 5:
            print(split_file)
            print(pack)
        alist, blist, gt = pair.split('\n')[1:-1]
        alist = np.array([int(a) for a in alist.split(' ')])
        blist = np.array([int(b) for b in blist.split(' ')])
        pairs.append((alist, blist, int(gt)))
    return pairs


def distance(disa, disb):
    from scipy.spatial.distance import cosine
    return cosine(disa, disb)


def get_sim(feat, pairs):
    sim = []
    for a, b, gt in pairs:
        fa = feat[a, :].mean(axis=0)
        fb = feat[b, :].mean(axis=0)
        sim.append((distance(fa, fb), gt))
    return np.array(sim)

def roc_acc(sim, errors):
    neg = sim[:, 1] == 0
    neg_sim = sim[neg, 0]
    pos_sim = sim[~neg, 0]
    sorted_neg = np.sort(neg_sim)
    n = sorted_neg.shape[0]
    acc = {}
    for e in errors:
        th = sorted_neg[int(np.ceil(n*e))]
        acc[e] = (pos_sim < th).mean()
    return acc


def verification(fc, split_files, errors):
    acc_list = []
    for split in split_files:
        pairs = load_pair(split)
        sim = get_sim(fc, pairs)
        acc_list.append(roc_acc(sim, errors))
    rst = []
    for e in errors:
        accs = np.array([a[e] for a in acc_list])
        rst.append((e, accs.mean(), accs.std()))
    return rst


def test_ijba(features):
    errors = [0.1, 0.01, 0.001]
    split_files = ['data/ijba/labels/idxs/split{}_idx.txt'.format(i+1) for i in range(10)]
    rst = verification(features, split_files, errors)
    return rst

def build_testset():
    with open(list_file, 'r') as f:
        lines = f.readlines()
        fns = [os.path.join(root, l.strip().split()[0]) for l in lines]
    return fns[:total]
