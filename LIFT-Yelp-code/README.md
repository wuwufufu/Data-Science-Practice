# Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts

This is the source code for the paper: [Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts](https://arxiv.org/abs/2309.10019) (ICML 2024).

## Requirements

* Python 3.8
* PyTorch 2.0
* Torchvision 0.15
* Tensorboard

- Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n lift python=3.8 -y
conda activate lift
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt
```

We encourage installing the latest dependencies. If there are any incompatibilities, please install the dependencies with the following versions.

```
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.1
yacs==0.1.8
tqdm==4.64.1
ftfy==6.1.1
regex==2022.7.9
timm==0.6.12
```

## Hardware

Most experiments can be reproduced using a single GPU with 20GB of memory (larger models such as ViT-L require more memory).

- To further reduce the GPU memory cost, gradient accumulation is recommended. Please refer to [Usage](#usage) for detailed instructions.

## Quick Start on the CIFAR-100-LT dataset

```bash
# run LIFT on CIFAR-100-LT (with imbalanced ratio=100)
python main.py -d cifar100_ir100 -m clip_vit_b16 adaptformer True
```

By running the above command, you can automatically download the CIFAR-100 dataset and run the method (LIFT).

## Running on Large-scale Long-tailed Datasets

### Prepare the Dataset

Download the dataset [Places](http://places2.csail.mit.edu/download.html), [ImageNet](http://image-net.org/index), and [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018).

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):

- Places

```
Path/To/Dataset
├─ train
│  ├─ airfield
|  |  ├─ 00000001.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ airfield
   |  ├─ Places365_val_00000435.jpg
   |  └─ ......
   └─ ......
```

- ImageNet

```
Path/To/Dataset
├─ train
│  ├─ n01440764
|  |  ├─ n01440764_18.JPEG
|  |  └─ ......
│  └─ ......
└─ val
   ├─ n01440764
   |  ├─ ILSVRC2012_val_00000293.JPEG
   |  └─ ......
   └─ ......
```

- iNaturalist 2018

```
Path/To/Dataset
└─ train_val2018
   ├─ Actinopterygii
   |  ├─ 2229
   |  |  ├─ 2c5596da5091695e44b5604c2a53c477.jpg
   |  |  └─ ......
   |  └─ ......
   └─ ......
```

### Reproduction

To reproduce the main result in the paper, please run

```bash
# run LIFT on ImageNet-LT
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True

# run LIFT on Places-LT
python main.py -d places_lt -m clip_vit_b16 adaptformer True

# run LIFT on iNaturalist 2018
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20
```

For other experiments, please refer to [scripts](scripts) for reproduction commands.

### Detailed Usage

To train and test the proposed method on more settings, run

```bash
python main.py -d [data] -m [model] [options]
```

The `[data]` can be the name of a .yaml file in [configs/data](configs/data), including `imagenet_lt`, `places_lt`, `inat2018`, `cifar100_ir100`, `cifar100_ir50`, `cifar100_ir10`, etc.

The `[model]` can be the name of a .yaml file in [configs/model](configs/model), including `clip_rn50`, `clip_vit_b16`, `in21k_vit_b16`, etc.

Note that using only `-d` and `-m` options denotes only fine-tuning the classifier. Please use additional `[options]` for more settings. 

- To apply lightweight fine-tuning methods, add options like `lora True`, `adaptformer True`, etc.

- To apply test-time ensembling, add `tte True`.

Moreover, `[options]` can facilitate modifying the configure options in [utils/config.py](utils/config.py). Following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.

- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.

- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.

- To apply gradient accumulation, add `micro_batch_size XX`. This can further reduce GPU memory costs. Note that `XX` should be a divisor of `batch_size`.

- To test an existing model, add `test_only True`. This option will test the model trained by your configure file. To test another model, add an additional option like `model_dir output/AnotherExpDir`.

- To test an existing model on the training set, add `test_train True`.

You can also refer to [scripts](scripts) for example commands.

## Acknowledgment

We thank the authors for the following repositories for code reference:
[[OLTR]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [[Classifier-Balancing]](https://github.com/facebookresearch/classifier-balancing), [[Dassl]](https://github.com/KaiyangZhou/Dassl.pytorch), [[CoOp]](https://github.com/KaiyangZhou/CoOp).

## Citation

If you find this repo useful for your work, please cite as:

```bibtex
@inproceedings{shi2024longtail,
  title={Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts},
  author={Jiang-Xin Shi and Tong Wei and Zhi Zhou and Jie-Jing Shao and Xin-Yan Han and Yu-Feng Li},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```


## Yelp VLM and TMR-LoRA Extensions

This repo now includes additional Yelp five-class experiments for the labels `food`, `drink`, `inside`, `outside`, and `menu`. The fixed split files are reused from `datasets/Yelp/Yelp_train.txt`, `Yelp_val.txt`, and `Yelp_test.txt`; no script below creates a new random split.

### Project Overview

A short code/data map is available at `docs/project_overview.md`. It describes the Yelp split files, caption files, class order, current CLIP/PEFT training path, existing analysis scripts, and where the new extension code lives.

### Unified Evaluation Outputs

New evaluation code writes a standard bundle under each experiment output directory:

```text
config.yaml
predictions.csv
metrics.json
classification_report.txt
confusion_matrix.csv
logits.npy              # when logits are available
parse_failures.csv      # for generative VLM parsing failures
```

For existing CLIP/PEFT training, `Trainer.test()` now saves these files in `eval_val/`, `eval_test/`, or `eval_train/`. Validation also appends per-epoch class-wise metrics to `eval_val/epoch_metrics.csv`.

### VLM Direct Baseline

Use `evaluate_vlm_direct.py` for direct generative VLM classification:

```bash
python evaluate_vlm_direct.py \
  --model qwen2_5_vl \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --split test \
  --mode image_caption_definition \
  --image_root /path/to/Yelp \
  --output_dir outputs/vlm_direct/qwen2_5_vl_image_caption_definition
```

Supported modes are `image_only`, `image_caption`, and `image_caption_definition`. The parser accepts outputs like `Food`, `food.`, or `The label is food`; unresolved responses are saved with `success=False` and `pred_label=unknown`.

If VLM dependencies are unavailable, run a smoke test with:

```bash
python evaluate_vlm_direct.py \
  --model mock \
  --split test \
  --max_samples 20 \
  --output_dir outputs/vlm_direct/mock_smoke
```

For Qwen2.5-VL, install a recent `transformers`, `accelerate`, and `torch` build that exposes `Qwen2_5_VLForConditionalGeneration`.

### Build VLM-LoRA Instruction Data

Convert the fixed Yelp splits to instruction tuning files:

```bash
python build_vlm_lora_dataset.py \
  --data_dir datasets/Yelp \
  --image_root /path/to/Yelp \
  --output_dir outputs/vlm_lora_dataset \
  --formats hf_jsonl,llava_json,qwen_jsonl
```

This writes files such as `train_hf.jsonl`, `val_hf.jsonl`, `test_hf.jsonl`, `train_llava.json`, and `train_qwen.jsonl`. Each sample contains image path, caption, instruction/question, and answer label.

### Train VLM-LoRA

`train_vlm_lora.py` provides a Qwen2.5-VL + PEFT LoRA entry point:

```bash
python train_vlm_lora.py \
  --model qwen2_5_vl \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl outputs/vlm_lora_dataset/train_hf.jsonl \
  --val_jsonl outputs/vlm_lora_dataset/val_hf.jsonl \
  --test_jsonl outputs/vlm_lora_dataset/test_hf.jsonl \
  --output_dir outputs/vlm_lora/qwen2_5_vl_r8 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --epochs 1 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --gradient_checkpointing
```

By default this applies LoRA to language-model projection/MLP modules. Add `--projector_lora` or pass `--target_modules` for custom module names. The script saves the adapter/checkpoint, training config, and generated test-set evaluation unless `--skip_eval` is set.

### Train Vanilla LoRA and TMR-LoRA

Use `train_clip_peft.py` for the CLIP image PEFT baselines requested by the Yelp experiments:

```bash
python train_clip_peft.py \
  --method vanilla_lora \
  --lora_rank 4 \
  --lora_alpha 16 \
  --classifier cosine \
  --loss CE \
  --output_dir outputs/tmr_lora/vanilla_lora_r4
```

TMR-LoRA oracle routing:

```bash
python train_clip_peft.py \
  --method tmr_lora \
  --routing oracle \
  --lora_rank 4 \
  --lora_alpha 16 \
  --classifier cosine \
  --loss CE \
  --output_dir outputs/tmr_lora/oracle_r4
```

Other routing modes:

```bash
python train_clip_peft.py --method tmr_lora --routing learned --lora_rank 4 --lora_alpha 16 --classifier cosine --loss CE --output_dir outputs/tmr_lora/learned_r4
python train_clip_peft.py --method tmr_lora --routing uniform --lora_rank 4 --lora_alpha 16 --classifier cosine --loss CE --output_dir outputs/tmr_lora/uniform_r4
python train_clip_peft.py --method tmr_lora --routing random  --lora_rank 4 --lora_alpha 16 --classifier cosine --loss CE --output_dir outputs/tmr_lora/random_r4
```

For logit adjustment, use `--loss "Logit Adjustment"` or `--loss LA`.

### Diagnostics

Analyze logits and learned routing utilization from a saved checkpoint:

```bash
python diagnose_tmr_lora.py \
  --method tmr_lora \
  --routing learned \
  --model-dir outputs/tmr_lora/learned_r4 \
  --mode test \
  --analysis-dir outputs/diagnostics/tmr_lora_learned_test
```

This saves a standard `eval_test/` bundle, `logit_distribution_summary.json`, and `expert_utilization.csv`. Add `--gradient-norm` to estimate class-wise gradient norms for LoRA/router parameters on a limited number of train batches.
