#!/bin/bash
work_path=$(dirname $0)
#while [ ! -f $work_path/checkpoints/ckpt_epoch_last.pth.tar ]; do
#    sleep 5m
#done
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p Spring_1080 -x BJ-IDC1-10-10-30-197 -n1 --gres=gpu:8 --ntasks-per-node=1 \
    python -u main.py \
    --config $work_path/config.yaml \
    --load-path $work_path/checkpoints/ckpt_epoch_11.pth.tar \
    --evaluate
