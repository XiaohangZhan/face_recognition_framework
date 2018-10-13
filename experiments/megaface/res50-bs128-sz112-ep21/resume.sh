#!/bin/bash
work_path=$(dirname $0)
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p Spring_face -n1 --gres=gpu:8 --ntasks-per-node=1 \
    python -u main.py \
    --config $work_path/config.yaml \
    --load-path $work_path/checkpoints/ckpt_epoch_5.pth.tar \
    --resume \
    --ngpu 8
