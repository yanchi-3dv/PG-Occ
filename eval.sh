#!/bin/bash
echo "preprocess data flag: "$IS_PREPROCESSING
echo "work_dir: "/output/${PNAME}/${JOBNAME}/${PODNAME}/output

echo "ENV: "${ENV}
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
if [ $NODE_RANK -eq 0 ]; then
    echo "number of nodes: "${NNODES}
fi
echo "launch node: "${NODE_RANK}

# opencv depend
echo "" > /etc/apt/sources.list

echo "
deb http://10.114.243.11:8888/ focal main restricted
deb http://10.114.243.11:8888/ focal-updates main restricted
deb http://10.114.243.11:8888/ focal universe
deb http://10.114.243.11:8888/ focal-updates universe
deb http://10.114.243.11:8888/ focal multiverse
deb http://10.114.243.11:8888/ focal-updates multiverse
deb http://10.114.243.11:8888/ focal-backports main restricted universe multiverse
deb http://10.114.243.11:8888/ focal-security main restricted
deb http://10.114.243.11:8888/ focal-security universe
deb http://10.114.243.11:8888/ focal-security multiverse
" | tee -a /etc/apt/sources.list > /dev/null

apt-get update
apt-get install -y libgl1
apt-get install -y wget
apt-get install -y libglm-dev
apt-get install -y libxrender1
DEBIAN_FRONTEND=noninteractive apt-get install -y libmagickwand-dev


export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# TRAIN
echo $ENV

get_value() {
    echo "$ENV" | awk -v key=$1 '{for(i=1;i<=NF;i++) if($i==key) print $(i+1)}'
}

weights=$(get_value --weights)

echo "Weights: $weights"
echo "Start Eval!!! Do not emo!!!"

cd /workspace/general_obstacle/
bash train_0.sh

# # check weights is path or url
# if [ -f "$weights" ]; then
#     echo "Weights is a file"
# else
#     echo "Weights is a url"
#     # download weights
#     wget $weights -O ckpts/latest_ckpt.pth
#     weights="ckpts/latest_ckpt.pth"
# fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: ${NUM_GPUS}"

export TORCH_CUDA_ARCH_LIST="8.0"

ZEEKR_WORK_DIR="/output/${PNAME}/${JOBNAME}/${PODNAME}/output"
echo "ZEEKR_WORK_DIR: ${ZEEKR_WORK_DIR}"
export ZEEKR_WORK_DIR=${ZEEKR_WORK_DIR}

ZEEKR_TB_DIR="/output/${PNAME}/${JOBNAME}/${PODNAME}/tensorboard"
echo "ZEEKR_TB_DIR: ${ZEEKR_TB_DIR}"
export ZEEKR_TB_DIR=${ZEEKR_TB_DIR}

# DATASET= ''
weights="https://oss-hz.zeekrlife.com/automl-perception-project/logs/zpilot_public/automl-zpilotpublic-train-pg-occ-debug-95koz/automl-zpilotpublic-train-pg-occ-debug-95koz-master-0/output/PGOcc/2025-08-28/07-02-28/epoch_12.pth?AWSAccessKeyId=CAF30C42A0BB55ACEF18&Expires=1759065321&Signature=xicIIqYRX3MPlwahLP%2B1P6%2Fbs%2B4%3D"
wget $weights -O ckpt/latest_ckpt.pth
weights="ckpt/latest_ckpt.pth"
sh scripts/val.sh $NUM_GPUS $weights ""
exit 0