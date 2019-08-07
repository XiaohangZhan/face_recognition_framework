import numpy as np
import torch
import logging
import io
import os
import sys
import pickle
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_state(state, path, epoch, is_last=False):
    assert path != ''
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(state, '{}_{}.pth.tar'.format(path, epoch))
    #if is_last:
        #os.system('cp {}_{}.pth.tar {}_last.pth.tar'.format(path, epoch, path))

def load_state(path, model, optimizer=None):
    if os.path.isfile(path):
        log("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        log("=> loaded checkpoint '{}' (epoch {} iteration {})".format(path, checkpoint['epoch'], checkpoint['count']))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return checkpoint
    else:
        log("=> no checkpoint found at '{}'".format(path))

def log(string):
    print(string)
    logging.info(string)


def bin_loader(path):
    '''load verification img array and label from bin file
    '''
    with open(path, 'rb') as f:
        if sys.version_info[0] == 2:
            data = pickle.load(open(path, 'rb'))
        elif sys.version_info[0] == 3:
            data = pickle.load(open(path, 'rb'), encoding='bytes')
        else:
            raise EnvironmentError('Only support python 2 or 3')
    bins, lbs = data
    assert len(bins) == 2*len(lbs)
    imgs = [pil_loader(b) for b in bins]
    return imgs, lbs

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
        return img


def normalize(feat, axis=1):
    if axis == 0:
        return feat / np.linalg.norm(feat, axis=0)
    elif axis == 1:
        return feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]
