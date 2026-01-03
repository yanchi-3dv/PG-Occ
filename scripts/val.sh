#!/bin/bash

GPU_NUM=$1
WEIGHTS=$2

if [ -z "$GPU_NUM" ]; then
    GPU_NUM=1
fi

python -m torch.distributed.launch --master_port=$((29500 + RANDOM % 100)) \
    --nproc_per_node=$GPU_NUM val.py  \
    --config configs/pgocc.py \
    --weights $WEIGHTS \
    --batch-size 1 \
