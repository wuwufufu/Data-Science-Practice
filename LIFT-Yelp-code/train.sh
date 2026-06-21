dataset=yelp_lt
model=clip_vit_b16
adaptformer=True
gpu=2
loss_type="LA"
la_tau=0.1
classifier="CosineClassifier"
optimizer="sgd"

nohup python main.py \
-d $dataset \
-m $model \
adaptformer $adaptformer \
gpu $gpu \
loss_type $loss_type \
la_tau $la_tau \
classifier $classifier \
optimizer $optimizer \
> train_$dataset\_$model\_$adaptformer\_$loss_type\_$la_tau\_$classifier\_$optimizer.log 2>&1 &
