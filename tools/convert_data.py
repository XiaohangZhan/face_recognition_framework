import os
import argparse
import pickle
import numpy as np
import mxnet as mx
import cv2
from tqdm import tqdm

def load_mx_rec(rec_path, save_path, write_img=True):
    if not os.path.isdir(save_path + "/images"):
        os.makedirs(save_path + "images")

    imgrec = mx.recordio.MXIndexedRecordIO(
        os.path.join(rec_path, 'train.idx'),
        os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    out_list = []
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        filename = "{}/{}_{}.jpg".format(label, label, idx)
        out_list.append("{} {}\n".format(filename, label))
        file_path = "{}/images/{}".format(save_path, label)
        if write_img:
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            cv2.imwrite('{}/images/{}'.format(save_path, filename), img)
    with open(os.path.join(save_path, "list.txt"), 'w') as f:
        f.writelines(out_list)


def load_bin(path, rootdir, image_size=[112,112]):
    if not os.path.isdir(rootdir + "/images"):
        os.makedirs(rootdir + "/images")
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite("{}/images/{}.jpg".format(rootdir, i), img)
    np.save('{}/issame_list.npy'.format(rootdir), np.array(issame_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rec_path",
                        help="mxnet record file path",
                        default='faces_emore', type=str)
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    load_mx_rec(args.rec_path, args.output_path, write_img=True)


if __name__ == "__main__":
    main()
