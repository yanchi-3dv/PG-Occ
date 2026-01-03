#!/bin/bash
GPU_NUM=$1
RESUME_FROM=$2

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

if [ -z "$RESUME_FROM" ]; then
    RESUME_FROM=None
fi

python -m torch.distributed.launch --master_port=$((29500 + RANDOM % 100)) \
    --nproc_per_node=$GPU_NUM train.py \
    --config configs/pgocc.py \
    --override debug=True \
    data.train.ann_file="data/nuscenes/nuscenes_infos_train_sweep.pkl"\
    data.val.ann_file="data/nuscenes/nuscenes_infos_val_sweep.pkl" \
    data.test.ann_file="data/nuscenes/nuscenes_infos_test_sweep.pkl" \
    resume_from=$RESUME_FROM