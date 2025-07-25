#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 scripts/train.py \
    task_name=${task_name} \
    task_config=${task_config} \
    expert_data_num=${expert_data_num} \
    seed=${seed} 