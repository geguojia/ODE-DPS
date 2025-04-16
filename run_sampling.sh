#/bin/bash

# $1: task
# $2: gpu number

python3 create_v.py

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/forward_process.yaml \
    --gpu=0 \
    --save_dir=myresult;
