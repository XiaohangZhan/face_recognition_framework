#!/bin/bash
work_path=$(dirname $0)
srun -p VI_ID_1080 -n1 --gres=gpu:8 --ntasks-per-node 8 \
    python -u main.py \
    --config $work_path/config.yaml
