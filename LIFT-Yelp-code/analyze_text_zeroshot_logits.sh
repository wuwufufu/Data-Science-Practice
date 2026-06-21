#!/usr/bin/env bash
set -euo pipefail

dataset="yelp_lt"
model="clip_vit_b16"
mode="test"
gpu="0"
env_name="imbclip"

/data00/jiahao/anaconda3/bin/conda run -n "${env_name}" \
python analyze_text_zeroshot_logits.py \
-d "${dataset}" \
-m "${model}" \
--mode "${mode}" \
gpu "${gpu}"
