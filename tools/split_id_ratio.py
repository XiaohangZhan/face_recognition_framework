import numpy as np
import os
import sys

def find_idx(base_id, d):
    ret = []
    for i in base_id:
        ret.extend(d[i])
    return ret

if __name__ == "__main__":

    list_fn = 'data/emore/list.txt'
    meta_fn = 'data/emore/meta.txt'
    split = [1./4, 1./4]  # ratios except for the last one.
                    # exampe: if ratios are 0.2, 0.3, 0.5, then split = [0.2, 0.3]
    output_prefix = 'data/emore_split112/'
    np.random.seed(0)

    ######### dont modify below #############
    if not os.path.isdir(os.path.dirname(output_prefix)):
        os.makedirs(os.path.dirname(output_prefix))
    with open(list_fn, 'r') as f:
        lines = f.readlines()
    with open(meta_fn, 'r') as f:
        lbs = f.readlines()
        cls = [int(l.strip()) for l in lbs[1:]]

    assert len(lines) == len(cls)

    d = dict()
    for i,c in enumerate(cls):
        if c in d:
            d[c].append(i)
        else:
            d[c] = [i]
    
    id_num = len(d.keys())
    keys = sorted(list(d.keys()))
    np.random.shuffle(keys)

    offset = 0
    for ii,sp in enumerate(split):
        split_num = int(sp * id_num)
        split_id = sorted(keys[offset:offset+split_num])
        offset += split_num
        split_mapping = dict(zip(split_id, range(split_num)))
        split_idx = find_idx(split_id, d)
        split_lines = [lines[i] for i in split_idx]
        split_labels_new = ['{}\n'.format(split_mapping[cls[i]]) for i in split_idx]
        with open(output_prefix + '{}_list.txt'.format(ii), 'w') as f:
            f.writelines(split_lines)
        with open(output_prefix + "{}_meta.txt".format(ii), 'w') as f:
            f.write('{} {}\n'.format(len(split_labels_new), len(split_id)))
            f.writelines(split_labels_new)

    remain_id = sorted(keys[offset:])
    remain_mapping = dict(zip(remain_id, range(len(remain_id))))
    remain_idx = find_idx(remain_id, d)
    remain_lines = [lines[i] for i in remain_idx]
    remain_labels_new = ['{}\n'.format(remain_mapping[cls[i]]) for i in remain_idx]
    with open(output_prefix + '{}_list.txt'.format(len(split)), 'w') as f:
        f.writelines(remain_lines)
    with open(output_prefix + '{}_meta.txt'.format(len(split)), 'w') as f:
        f.write('{} {}\n'.format(len(remain_labels_new), len(remain_id)))
        f.writelines(remain_labels_new)
    # check
#    if True:
#        num_images = []
#        num_ids = []
#        for ii in range(len(split)+1):
#            with open(inputfn[:-4] + '_{}'.format(ii) + '.txt', 'r') as f:
#                check_lines = f.readlines()
#            num_images.append(len(check_lines))
#            num_ids.append(int(check_lines[-1].strip().split(' ')[-1])+1)
#        pdb.set_trace()
#        assert sum(num_images) == len(lines)
#        assert sum(num_ids) == id_num
