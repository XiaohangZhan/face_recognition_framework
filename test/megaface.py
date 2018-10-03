import os, sys
import numpy as np
import pdb
import time
import multiprocessing as mp

sys.path.append('/mnt/lustre/zhouyucong/infrastructure/faiss_py3')
import faiss

from .compute_roc import compute_roc_points

__all__ = ["test_megaface", "test_megaface_roc"]

probe_num = 3530
label_file = 'data/megaface/facescrub3530/image_list.label.meta'
# probe_num = 3964
# label_file = '/mnt/lustre/zhouyucong/verify/testsets/train/megaface_probe_3964/image_list.label.meta'
# distractor_num = 992919

def normalize(vec):
    for i in range(vec.shape[0]):
        vec[i] = vec[i] / np.linalg.norm(vec[i])
    return vec

class MegaFaceTest:
    def __init__(self, probe_feature, probe_label, distractor_feature = None, top_ks = [1]):
        # Read probe feature and labels
        self.feature_dim = probe_feature.shape[1]
        self.probe_feature = normalize( probe_feature )
        self.probe_num = probe_feature.shape[0]
        self.probe_label = probe_label.reshape( self.probe_num )
        self.top_ks = top_ks
        assert probe_label.shape[0] == self.probe_num, "Probe_feature and probe_label include different numbers of images!"
        # Read distractor feature and labels
        if type(distractor_feature) != type(None):
            assert probe_feature.shape[1] == self.feature_dim, "Probe_feature and distractor_feature have different dims of feature!"
            self.distractor_feature = normalize( distractor_feature )
            self.distractor_num = distractor_feature.shape[0]
            # print "Distractor set contains %d images."%(self.distractor_num)
        else:
            self.distractor_num = 0
            # print "Doesn't have distractor set. Will do test on probe set only!"
        # Calculate similarity
        self.self_similarity = self.probe_feature.dot( self.probe_feature.T )
        self.self_similarity[ np.arange(self.probe_num), np.arange(self.probe_num) ] = -1
        if self.distractor_num:
            self.cross_similarity = self.probe_feature.dot( self.distractor_feature.T )

    def identification(self):
        # print "Calculating MegaFace identification!"
        if self.distractor_num == 0:
            print("Doesn't have distractor set. Can't do MegeFace identification test. Please try normal identification!")
            return 0
        rst = {}
        for top_k in self.top_ks:
            hit = 0; pair = 0
            for label in set( self.probe_label ):
                idx = np.nonzero( self.probe_label == label )[0]
                num = idx.shape[0]
                temp_self = self.self_similarity[ np.ix_(idx, idx) ]
                temp_max = np.tile( self.cross_similarity[idx, :].max(1).reshape(num, 1), (1, num) )
                hit += np.sum( temp_self > temp_max )
                pair += num * (num - 1)
            rst[top_k] = float(hit) / pair
        # return {'top_{}'.format(k): rst[k] for k in self.top_ks}
        return [rst[k] for k in self.top_ks]

    def verification(self):
        print("Calculating verification!")
        # prepare probe set self label
        temp_label = np.zeros( (self.probe_num, self.probe_num), np.int )
        for label in set( self.probe_label ):
            idx = np.nonzero( self.probe_label == label )[0]
            num = idx.shape[0]
            temp_label[ np.ix_(idx, idx) ] = 1
        self_label = temp_label[ np.tril_indices(self.probe_num, -1) ]
        self_score = self.self_similarity[ np.tril_indices(self.probe_num, -1) ]
        # If have distractor set, prepare cross label
        if self.distractor_num:
            cross_label = np.zeros( (self.probe_num * self.distractor_num), np.int )
            cross_score = self.cross_similarity.reshape( self.probe_num * self.distractor_num )
            labels = np.hstack( (self_label, cross_label) )
            scores = np.hstack( (self_score, cross_score) )
        else:
            labels = self_label
            scores = self_score
        # return fpr, tpr, threshold
        print('labels: ', labels)
        print('scores: ', scores)
        print(len(labels), len(scores))




def worker(feat, topk_score, idx, cos_margin):

    topk_score = topk_score.reshape(-1, 1)

    n = topk_score.shape[0]

    score = feat.dot(feat.T)

    tf = score > topk_score + cos_margin
    hit = np.sum(tf, axis=1) - 1
    hit = np.sum(hit)

    hit_idx = []
    hit_score = []
    miss_idx = []
    miss_score = []
    for i in range(n):
        tmp_hit = np.where(tf[i,:])[0]
        tmp_miss = np.where(np.logical_not(tf[i,:]))[0]
        hit_idx.append(idx[tmp_hit])
        hit_score.append(score[i,tmp_hit])
        miss_idx.append(idx[tmp_miss])
        miss_score.append(score[i,tmp_miss])

    return hit, n * (n-1), hit_idx, hit_score, miss_idx, miss_score


#def compute_rank(feat_prob, feat_distractor, labels, n_dist, cos_margin):


def compute_rank_faiss(feat_prob, feat_distractor, labels, n_dist, cos_margin):

    feat_dim = feat_prob.shape[1]

    #ngpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    ngpu = 2
    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpu)]
    indexes = [faiss.GpuIndexFlatIP(res[i], feat_dim, flat_config[i]) for i in range(ngpu)]
    index = faiss.IndexProxy()
    for sub_index in indexes:
        index.addIndex(sub_index)

    max_query = 1024
    label_map = {}
    for i,l in enumerate(labels):
        if l in label_map:
            label_map[l].append(i)
        else:
            label_map[l] = [i]

    index.reset()
    index.add(feat_distractor)
    n_dist = min(n_dist, feat_distractor.shape[0])

    hit = 0
    n = 0
    hit_idx = []
    hit_score = []
    miss_idx = []
    miss_score = []
    dis_idx = []
    dis_score = []
    for l, idx in label_map.items():
        idx = np.array(idx)
        assert len(idx) < max_query, 'number of images of label {} ({}) > {}'.format(l, len(idx), max_query)
        this_feat = feat_prob[idx,:]
        this_topk_score, this_topk_idx = index.search(this_feat, n_dist)

        this_hit, this_n, a, b, c, d = worker(this_feat, this_topk_score[:,0], idx, cos_margin)
        hit += this_hit
        n += this_n
        hit_idx.extend(a)
        hit_score.extend(b)
        miss_idx.extend(c)
        miss_score.extend(d)
        dis_idx.extend(list(map(list, this_topk_idx)))
        dis_score.extend(list(map(list, this_topk_score)))

    return hit / n, hit_idx, hit_score, miss_idx, miss_score, dis_idx, dis_score


def test_megaface(features):
    prob_feat = features[:probe_num, :]
    dist_feat = features[probe_num:, :]

    with open(label_file, 'r') as f:
        lines = f.readlines()
        num_img, num_id = map(int, lines[0].rstrip().split())
        labels = [int(x.strip()) for x in lines[1:]]
        prob_label = np.array(labels)

    results = []
    for dis_size in [10000,100000, 1000000]:
        # with open('../msics/id_%d_clean'%dis_size) as f:
        with open('data/megaface/megaface_distractor/id_%d_clean'%dis_size) as f:
            lines = f.readlines()
            lines = [int(x.strip())-1 for x in lines]
            idx = np.array(lines)

            megaface = MegaFaceTest(prob_feat, prob_label, dist_feat[idx, :], [1] )
            res = megaface.identification()
            results.append(res[0])

    return results

def test_megaface_faiss(features):
    probe_feat = features[:probe_num, :]
    dist_feat = features[probe_num:, :]
    with open(label_file, 'r') as f:
        lines = f.readlines()
        num_img, num_id = map(int, lines[0].rstrip().split())
        labels = [int(x.strip()) for x in lines[1:]]
        prob_label = np.array(labels)

    results = []
    for dis_size in [10000, 100000, 1000000]:
        with open('../msics/id_%d_clean'%dis_size) as f:
            print('testing: distractor %d:...'%len(idx), end='')
            lines = f.readlines()
            lines = [int(x.strip())-1 for x in lines]
            idx = np.array(lines)
            t0 = time.time()
            res, hit_idx, hit_score, miss_idx, miss_score, dis_idx, dis_score = \
            compute_rank(prob_feat, dist_feat[idx,:], labels, n_dist, cos_margin)
            t1 = time.time()
            print('top-1=%.2f, in %fs'%(res*100, t1-t0))
            results.append(res)

    return results



def roc_evaluation(feats, labels, dist_feats):

    fpr_points = [-1,-2,-3,-4,-5,-6,-7]
    fprs = [10**p for p in fpr_points]
    feat_num = feats.shape[0]
    dist_num = dist_feats.shape[0]

    ## simialrity between probe and gallery
    self_similarity = feats.dot(feats.T)
    self_similarity[np.arange(feat_num), np.arange(feat_num)] = -1
    ## similairty between probe and probe
    cross_similarity = feats.dot(dist_feats.T)

    all_indices = np.arange(0, feat_num)
    allscores = []
    alllabels = []
    for l in range(feat_num):
        this_prb_feat = feats[l, :]
        this_prb_meta = labels[l]

        this_indices = np.concatenate((all_indices[:l], all_indices[(l+1):]))
        this_self_labels = labels[this_indices]
        this_self_similarity = self_similarity[l, this_indices]
        this_cross_similarity  = cross_similarity[l, :]

        allscores.append(np.concatenate((this_self_similarity, this_cross_similarity), axis=0))
        this_all_labels = np.concatenate((this_self_labels, np.zeros(dist_num)-1), axis=0)
        alllabels.append((this_all_labels == this_prb_meta).astype(np.int))

    allscores = np.concatenate(allscores, axis=0)
    alllabels = np.concatenate(alllabels, axis=0)

    tpr_k_score, th_k_score, tpr_sp = compute_roc_points(alllabels, allscores, fprs)


    print("fpr    | "+" | ".join('{:^5}'.format(i) for i in fpr_points)+" |")
    print("|".join("  :-:  " for i in range(len(fpr_points)+1))+"|")
    print("tpr(%) | "+" | ".join('{:5.2f}'.format(i*100) for i in tpr_k_score)+" |")
    print("thres  | "+" | ".join('{:5.3f}'.format(i) for i in th_k_score)+" |")

    return tpr_k_score

def test_megaface_roc(features):
    prob_feat = features[:probe_num, :]
    dist_feat = features[probe_num:, :]
    with open(label_file, 'r') as f:
        lines = f.readlines()
        num_img, num_id = map(int, lines[0].rstrip().split())
        labels = [int(x.strip()) for x in lines[1:]]
        prob_label = np.array(labels)

    results = []
    #for dis_size in [10000, 100000, 1000000]:
    dis_size = 100000
    # with open('../msics/id_100000_clean'%dis_size) as f:
    with open('/mnt/lustrenew/face/DATA/megaface_test/megaface_distractor/idx_%d.txt'%dis_size) as f:
        lines = f.readlines()
        lines = [int(x.strip())-1 for x in lines]
        idx = np.array(lines)
        tpr = roc_evaluation(prob_feat, prob_label, dist_feat[idx, :])
        results = tpr[-3:]

    return results

