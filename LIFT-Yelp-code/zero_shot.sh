dataset=yelp_lt
model=clip_vit_b16
zero_shot=True

export CUDA_VISIBLE_DEVICES=1
nohup python main.py \
-d $dataset \
-m $model \
zero_shot $zero_shot \
> zero_shot_$dataset_$model_$zero_shot.log 2>&1 &