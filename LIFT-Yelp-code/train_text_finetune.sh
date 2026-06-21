#!/usr/bin/env bash
set -euo pipefail

dataset=yelp_lt
model=clip_vit_b16
text_finetune=True
adaptformer=True
gpu=0
loss_type="LA"
classifier="LinearClassifier"
optimizer="sgd"
la_tau=0.25

nohup python main.py \
-d $dataset \
-m $model \
text_finetune $text_finetune \
adaptformer $adaptformer \
gpu $gpu \
loss_type $loss_type \
classifier $classifier \
optimizer $optimizer \
la_tau $la_tau \
> train_text_ft_${dataset}_${model}_${adaptformer}_${loss_type}_${la_tau}_${classifier}_${optimizer}.log 2>&1 &
