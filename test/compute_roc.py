import numpy as np
import time

def compute_roc_points(labels, scores, fprs, use_sklearn=True):
    tpr_k_score = []
    th_k_score = []
    sp_tpr = 0
    print(labels.shape)
    print(scores.shape)


    if use_sklearn:
        from sklearn.metrics import roc_curve
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)

        sp_idx = np.argmin(np.abs(roc_tpr+roc_fpr-1))
        sp_tpr = roc_tpr[sp_idx]
        for fpr_ratio in fprs:
            idx = np.argmin(np.abs(roc_fpr - fpr_ratio))
            tpr = roc_tpr[idx]
            th = roc_thresholds[idx]
            tpr_k_score.append(tpr)
            th_k_score.append(th)

        return tpr_k_score, th_k_score, sp_tpr

    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    cum_pos = np.cumsum(sorted_labels, dtype=float)
    t4 = time.time()
    total_pos = cum_pos[-1]
    n = labels.size
    fn = cum_pos - sorted_labels
    tp = total_pos - fn
    fp = np.arange(n,0,-1) - tp
    t5 = time.time()

    tpr = tp/total_pos
    fpr = fp/(n-total_pos)

    sp_idx = np.argmin(np.abs(tpr+fpr-1))

    for fp in fprs:
        idx = np.argmin(np.abs(fpr-fp))
        tpr_k_score.append(tpr[idx])
        th_k_score.append(sorted_scores[idx])
        # print("%6f %7f %6f" % (tpr[idx], fpr[idx], sorted_scores[idx]))

    # print("%6f"%tpr[sp_idx])
    return tpr_k_score, th_k_score, tpr[sp_idx]

def compute_roc_part(worker_id, feat1, feat2, meta1, meta2, delta, thres, tp, fp, total_pos_neg):
    scores = feat1.dot(feat2.T)
    labels = (meta1.reshape(-1,1) == meta2.reshape(1,-1)).astype(np.int)
    if delta != -1:
        indices = np.triu_indices(delta, k=1)
        scores = scores[indices]
        labels = labels[indices]
    else:
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    cum_pos = np.cumsum(sorted_labels, dtype=float)
    total_pos = cum_pos[-1]
    n = labels.size
    fn = cum_pos - sorted_labels
    tp_tmp = total_pos - fn
    fp_tmp = np.arange(n, 0, -1) - tp_tmp
    import bisect
    c_tp = [0]*len(thres)
    c_fp = [0]*len(thres)
    start = 0
    for i, th in enumerate(thres):
        #'Find rightmost value less than or equal to x'
        pos = bisect.bisect_right(sorted_scores, th, start)
        if pos != len(sorted_scores):
            c_tp[i] = tp_tmp[pos]
            c_fp[i] = fp_tmp[pos]
            start = pos
        else:
            c_tp[i] = total_pos
            c_fp[i] = 0
    total_pos_neg[worker_id] = np.array([total_pos, n - total_pos])
    tp[worker_id] = c_tp
    fp[worker_id] = c_fp

