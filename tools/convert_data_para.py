import os
import argparse
import pickle
import numpy as np
import mxnet as mx
import cv2
import multiprocessing as mp
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rec_path",
                    help="mxnet record file path",
                    default='faces_emore', type=str)
parser.add_argument("-o", "--output_path", type=str)
args = parser.parse_args()

rec_path = args.rec_path
save_path = args.output_path

if not os.path.isdir(save_path + "/images"):
    os.makedirs(save_path + "images")
imgrec = mx.recordio.MXIndexedRecordIO(
    os.path.join(rec_path, 'train.idx'),
    os.path.join(rec_path, 'train.rec'), 'r')

def task(idx):
    img_info = imgrec.read_idx(idx)
    header, img = mx.recordio.unpack_img(img_info)
    label = int(header.label)
    filename = "{}/{}_{}.jpg".format(label, label, idx)
    ret = "{} {}\n".format(filename, label)
    cv2.imwrite('{}/images/{}'.format(save_path, filename), img)
    return ret

img_info = imgrec.read_idx(0)
header,_ = mx.recordio.unpack(img_info)
max_idx = int(header.label[0])
count = max_idx - 1

img_info_last = imgrec.read_idx(max_idx - 1)
header_last,_ = mx.recordio.unpack(img_info_last)
max_label = int(header_last.label)

# mkdir
for i in range(max_label + 1):
    if not os.path.isdir("{}/images/{}".format(save_path, i)):
        os.makedirs("{}/images/{}".format(save_path, i))

pool = mp.Pool(mp.cpu_count())
out_list = list(tqdm(pool.imap(task, range(1, max_idx)), total=count))
#out_list = [task(i) for i in range(1, max_idx)]
with open(os.path.join(save_path, "list.txt"), 'w') as f:
    f.writelines(out_list)
