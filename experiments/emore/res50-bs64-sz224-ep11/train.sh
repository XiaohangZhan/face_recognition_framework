#!/bin/bash
work_path=$(dirname $0)
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p Spring_1080 -n1 --gres=gpu:8 --ntasks-per-node=1 \
    python -u main.py \
    --config $work_path/config.yaml
