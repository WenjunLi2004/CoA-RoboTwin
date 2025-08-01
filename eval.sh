#!/bin/bash

policy_name=CoA 
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    # [TODO] add parameters here