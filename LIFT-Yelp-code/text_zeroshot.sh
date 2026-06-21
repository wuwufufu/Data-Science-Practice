#!/usr/bin/env bash
set -euo pipefail

dataset=yelp_lt
model=clip_vit_b16
text_zeroshot=True
gpu=0

export CUDA_VISIBLE_DEVICES=${gpu}
nohup python main.py \
-d ${dataset} \
-m ${model} \
text_zeroshot ${text_zeroshot} \
> text_zeroshot_${dataset}_${model}_${text_zeroshot}.log 2>&1 &

echo "Launched text zero-shot run on GPU ${gpu}."
