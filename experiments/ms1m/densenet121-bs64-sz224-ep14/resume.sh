#!/bin/bash
work_path=$(dirname $0)
epoch=$1
python -u main.py \
    --config $work_path/config.yaml \
    --load-path $work_path/checkpoints/ckpt_epoch_${epoch}.pth.tar \
    --resume
