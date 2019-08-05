import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import io
import test
from PIL import Image
import torchvision.transforms as transforms
try:
    import mc
except ImportError:
    pass
from utils import bin_loader

import pdb

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

class FaceDataset(Dataset):
    def __init__(self, config, task_idx, phase):
        self.root_dir = config.train.data_root[task_idx]
        self.config = config
        assert phase in ['train', 'val', 'test', 'extract']
        self.phase = phase
        
        if phase in ['train', 'val']:
            print("Building task #{} dataset from {} and {}".format(task_idx, config.train.data_list[task_idx], config.train.data_meta[task_idx]))
            with open(config.train.data_list[task_idx], 'r') as f:
                lines = f.readlines()
                self.lists = [os.path.join(config.train.data_root[task_idx], l.strip()) for l in lines]
            with open(config.train.data_meta[task_idx], 'r') as f:
                lines = f.readlines()
                num_img, num_class = lines[0].strip().split()
                self.num_img, self.num_class = int(num_img), int(num_class)
                self.metas = [int(l.strip()) for l in lines[1:]]
            assert self.num_img == len(self.lists)
            assert self.num_img == len(self.metas)
            assert self.num_class > max(self.metas)
        elif phase == "test": # test
            if config.test.benchmark == "megaface":
                self.lists = test.megaface.build_testset()
            elif config.test.benchmark == "ijba":
                self.lists = test.ijba.build_testset()
            elif config.test.benchmark == 'lfw':
                self.lists = test.lfw.build_testset()
            else:
                raise Exception("No such benchmark: {}".format(config.test.benchmark))
            self.num_img = len(self.lists)
            self.metas = None
        else: # extract
            with open(config.extract_info.data_list, 'r') as f:
                lines = f.readlines()
                self.lists = [os.path.join(config.extract_info.data_root, l.strip()) for l in lines]
            self.num_img = len(self.lists)
            self.metas = None

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3125, 0.3125, 0.3125])
        self.transforms = transforms.Compose([transforms.ToTensor(), normalize])
        self.initialized = False
 
    def __len__(self):
        return self.num_img
 
    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.config.memcached_client)
            client_config_file = "{}/client.conf".format(self.config.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read_one(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_img)
        filename = self.lists[idx]
        if self.metas is not None:
            label = self.metas[idx]
        else:
            label = 0
        try:
            value = mc.pyvector()
            self.mclient.Get(filename, value)
            value_str = mc.ConvertBuffer(value)
            img = pil_loader(value_str)
        except:
            print('Read image[{}] failed ({})'.format(idx, filename))
            return self._read_one()
        else:
            return img, label
            
    def __getitem__(self, idx):
        self.__init_memcached()
        ## memcached
        if self.config.memcached:
            img, label = self._read_one(idx)
        else:
            filename = self.lists[idx]
            if self.metas is not None:
                label = self.metas[idx]
            else:
                label = 0
            if not os.path.isfile(filename):
                raise Exception('Read image[{}] failed ({})'.format(idx, filename))
            img = Image.open(filename).convert('RGB')

        ## transform & aug
        if self.phase == 'train' and self.config.train.augmentation['flip_aug']:
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.phase == 'train':
            scale_height_diff = (np.random.rand() * 2 - 1) * self.config.train.augmentation['scale_aug']
            scale_width_diff = (np.random.rand() * 2 - 1) * self.config.train.augmentation['scale_aug']
            trans_diff_x = (np.random.rand() * 2 - 1) * self.config.train.augmentation['trans_aug']
            trans_diff_y = (np.random.rand() * 2 - 1) * self.config.train.augmentation['trans_aug']
        else:
            scale_height_diff = 0.
            scale_width_diff = 0.
            trans_diff_x = 0.
            trans_diff_y = 0.

        crop_height_aug = self.config.transform.crop_size * (1 + scale_height_diff)
        crop_width_aug = self.config.transform.crop_size * (1 + scale_width_diff)
        center = (img.width / 2. * (1 + trans_diff_x), (img.height / 2. + self.config.transform.crop_center_y_offset) * (1 + trans_diff_y))

        if center[0] < crop_width_aug / 2:
            crop_width_aug = center[0] * 2 - 0.5
        if center[1] < crop_height_aug / 2:
            crop_height_aug = center[1] * 2 - 0.5
        if center[0] + crop_width_aug / 2 >= img.width:
            crop_width_aug = (img.width - center[0]) * 2 - 0.5
        if center[1] + crop_height_aug / 2 >= img.height:
            crop_height_aug = (img.height - center[1]) * 2 - 0.5

        rect = (center[0] - crop_width_aug / 2, center[1] - crop_height_aug / 2,
                center[0] + crop_width_aug / 2, center[1] + crop_height_aug / 2)
        img = img.crop(rect)
        img = img.resize((self.config.transform.final_size, self.config.transform.final_size), Image.BICUBIC)

        if False: #DEBUG
            img.save("output/{}.jpg".format(idx))

        img = self.transforms(img)
        return img, label

class GivenSizeSampler(Sampler):
    '''
    Sampler with given total size
    '''
    def __init__(self, dataset, total_size=None, rand_seed=None, sequential=False):
        self.rand_seed = rand_seed if rand_seed is not None else 0
        self.dataset = dataset
        self.epoch = 0
        self.sequential = sequential
        self.total_size = total_size if total_size is not None else len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.sequential:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.rand_seed)
            origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            origin_indices = list(range(len(self.dataset)))
        indices = origin_indices[:]

        # add extra samples to meet self.total_size
        extra = self.total_size - len(origin_indices)
        print('Origin Size: {}\tAligned Size: {}'.format(len(origin_indices), self.total_size))
        if extra < 0:
            indices = indices[:self.total_size]
        while extra > 0:
            intake = min(len(origin_indices), extra)
            indices += origin_indices[:intake]
            extra -= intake
        assert len(indices) == self.total_size, "{} vs {}".format(len(indices), self.total_size)

        return iter(indices)

    def __len__(self):
        return self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class BinDataset(Dataset):
    def __init__(self, bin_file, transform=None):
        self.img_lst, _ = bin_loader(bin_file)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)
        try:
            img = self.img_lst[idx]
            return img
        except Exception as err:
            print('Read image[{}] failed ({})'.format(idx, err))
            return self._read()

    def __getitem__(self, idx):
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_labeled_dataset(filelist, prefix):
    img_lst = []
    lb_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            n, lb = x.strip().split(' ')
            lb = int(lb)
            img_lst.append(os.path.join(prefix, n))
            lb_lst.append(lb)
    assert len(img_lst) == len(lb_lst)
    return img_lst, lb_lst

def build_unlabeled_dataset(filelist, prefix):
    img_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            img_lst.append(os.path.join(prefix, x.strip().split(' ')[0]))
    return img_lst


class FileListLabeledDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst, self.lb_lst = build_labeled_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform
        self.num_class = max(self.lb_lst) + 1
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        lb = self.lb_lst[idx]
        try:
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            else:
                img = pil_loader(open(fn, 'rb').read())
            return img, lb
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img, lb = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb

class FileListDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst = build_unlabeled_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        try:
            #img = pil_loader(open(fn, 'rb').read())
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            else:
                img = pil_loader(open(fn, 'rb').read())
            return img
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img
