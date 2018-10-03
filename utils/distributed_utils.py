import os
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data.sampler import Sampler
import math
import numpy as np
import pdb
from .common_utils import log
import multiprocessing as mp

class DistModule(Module):
    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

def average_gradients(model):
    """ average gradients """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)

def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def dist_init(port):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    print(addr)

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


#class DistModule(Module):
#    def __init__(self, module):
#        super(DistModule, self).__init__()
#        self.module = module
#        broadcast_params(self.module)
#        dist._clear_group_cache()
#    def forward(self, *inputs, **kwargs):
#        return self.module(*inputs, **kwargs)
#    def train(self, mode=True):
#        dist._clear_group_cache()
#        super(DistModule, self).train(mode)
#        self.module.train(mode)
#
#def average_gradients(model):
#    """ average gradients """
#    for param in model.parameters():
#        if param.requires_grad:
#            dist.all_reduce(param.grad.data)
#
#def broadcast_params(model):
#    """ broadcast model parameters """
#    for p in model.state_dict().values():
#        dist.broadcast(p, 0)
#
#def dist_init(port):
#    proc_id = int(os.environ['SLURM_PROCID'])
#    ntasks = int(os.environ['SLURM_NTASKS'])
#    node_list = os.environ['SLURM_NODELIST']
#    num_gpus = torch.cuda.device_count()
#    torch.cuda.set_device(proc_id%num_gpus)
#
#    if '[' in node_list:
#        beg = node_list.find('[')
#        pos1 = node_list.find('-', beg)
#        if pos1 < 0:
#            pos1 = 1000
#        pos2 = node_list.find(',', beg)
#        if pos2 < 0:
#            pos2 = 1000
#        node_list = node_list[:min(pos1,pos2)].replace('[', '')
#    addr = node_list[8:].replace('-', '.')
#    print(addr)
#
#    os.environ['MASTER_PORT'] = port
#    os.environ['MASTER_ADDR'] = addr
#    os.environ['WORLD_SIZE'] = str(ntasks)
#    os.environ['RANK'] = str(proc_id)
#    dist.init_process_group(backend='nccl')
#
#    rank = dist.get_rank()
#    world_size = dist.get_world_size()
#    return rank, world_size

class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        assert len(self.dataset) > self.world_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGivenSizeSampler(Sampler):
    '''
    Sampler with given total size
    '''
    def __init__(self, dataset, total_size=None, world_size=None, rank=None, rand_seed=None):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.rand_seed = rand_seed if rand_seed is not None else 0
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        if total_size is None:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        else:
            self.num_samples = int(math.ceil(total_size * 1.0 / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rand_seed)
        origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        indices = origin_indices[:]

        # add extra samples to meet self.total_size
        extra = self.total_size - len(origin_indices)
        if self.rank == 0:
            log('Origin Size: {}    Modified Size: {}'.format(len(origin_indices), self.total_size))
        if extra < 0:
            indices = indices[:self.total_size]
        while extra > 0:
            intake = min(len(origin_indices), extra)
            indices += origin_indices[:intake]
            extra -= intake
        assert len(indices) == self.total_size, "{} vs {}".format(len(indices), self.total_size)

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def gather_tensors(input_array):
    world_size = dist.get_world_size()
    ## gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)]
    dist.all_gather(all_shape, shape_tensor)
    ## compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    ## padding tensors and gather them
    output_tensors = [torch.Tensor(max_count).cuda() for i in range(world_size)]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)
    ## unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [x[:all_count[i]].reshape(all_shape[i]) for i,x in enumerate(padded_output)]
    return output
