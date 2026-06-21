# Yelp 五分类项目理解报告

## 数据组织

项目沿用 LIFT/long-tail classification 的目录结构，Yelp 相关元数据位于 `datasets/Yelp/`：

- `Yelp_train.txt`、`Yelp_val.txt`、`Yelp_test.txt`：固定 train/val/test split。每行格式为 `relative_image_path label_id`，例如 `yelp_filtered_image/xxx.jpg 1`。
- `Yelp_train_text.txt`、`Yelp_val_text.txt`、`Yelp_test_text.txt`：caption/title 文件。每行以 tab 分隔，格式为 `photo_id caption label_id`，caption 可能为空。
- `classnames.txt`：类别 id 到类别名的映射。当前顺序是 `0 drink`、`1 food`、`2 inside`、`3 menu`、`4 outside`。
- `split_stats.json`、`invalid_images.txt`：数据划分统计和无效图片记录。

图片根目录由 `configs/data/yelp_lt.yaml` 的 `root` 指定，当前配置为 `/data00/zhiyuan_huang/datasets/Yelp`。split 文件中的相对路径会拼到该 root 后面，例如 `<root>/yelp_filtered_image/xxx.jpg`。

数据集实现位于 `datasets/yelp_lt.py`：

- `Yelp_LT` 读取图片和标签，用于普通 image classification。
- `Yelp_MM_LT` 额外按 `photo_id` 读取 caption，用于 text zero-shot / text fine-tuning。该类会丢弃 caption 为空的样本，并打印过滤前后的各类计数。
- 底层读取逻辑来自 `datasets/lt_data.py` 的 `LT_Dataset`。

## 当前模型与训练脚本

主入口是 `main.py`。它读取 `configs/data/*.yaml` 和 `configs/model/*.yaml`，合并命令行 `opts` 后创建 `Trainer`。

核心训练逻辑位于 `trainer.py`：

- `Trainer.build_data_loader()` 构建 train、train_init、train_test、val、test dataloader，并固定使用已有 split 文件。
- `Trainer.build_model()` 根据配置构建 CLIP zero-shot、CLIP image PEFT、CLIP text PEFT 或 IN21K ViT PEFT。
- `Trainer.train()` 训练 image PEFT/classifier，按 val macro-F1 保存最佳 `checkpoint.pth.tar`。
- `Trainer.train_text()` 训练 caption/text encoder 分支。
- `Trainer.test()` 在 train/val/test 上评估，当前主要通过 `utils/evaluator.py` 打印 accuracy、macro-F1、class acc、worst-case acc、many/medium/few acc 等。

模型相关代码集中在 `models/`：

- `models/models.py`：封装 `ZeroShotCLIP`、`PeftModelFromCLIP`、`PeftModelFromViT`、`PeftTextModelFromCLIP`。
- `models/peft_vit.py`：ViT/CLIP-ViT 图像分支 PEFT 注入与 forward。已有 LoRA 注入 attention q/v，可选 `lora_mlp` 注入 MLP。
- `models/peft_text.py`：CLIP text encoder 的 PEFT 注入。
- `models/peft_modules.py`：已有 `VPT`、`Adapter`、`AdaptFormer`、`LoRA`、`SSF`、`MaskedLinear` 等轻量模块。
- `models/classifiers.py`：`LinearClassifier`、`CosineClassifier`、`L2NormedClassifier`、`LayerNormedClassifier`。

常用训练脚本包括：

- `train.sh`：基础训练示例。
- `train_peft_modules.sh`：批量跑 Adapter/AdaptFormer/LoRA/VPT/SSF/full tuning 等 PEFT 对比。
- `train_text_finetune.sh`、`text_zeroshot.sh`、`zero_shot.sh`：文本或 zero-shot 相关实验。

## 当前结果与分析输出

原始训练会在 `output/<run_name>/` 下保存：

- `log.txt`：由 `utils/logger.py` 重定向的训练日志。
- `tensorboard/`：训练曲线。
- `checkpoint.pth.tar`：最佳 val macro-F1 checkpoint，包含 `tuner` 和 `head` state dict。

已有分析脚本包括：

- `analyze_image_logits.py`：加载 image checkpoint，统计正确/错误样本 top-1 logit 分布，保存 `summary.json`、`raw_logits.npz` 和图。
- `analyze_text_zeroshot_logits.py`：分析 text zero-shot 或 text fine-tune logits。
- `fuse_image_text_logits.py`：收集 image/text logits，支持在 val 上搜索 fusion alpha 后测 test。
- `visualize_dataset.py`：生成 Yelp 数据分布和 caption 缺失统计图。

原有 `Trainer.test()` 目前不统一保存逐样本预测、classification report、confusion matrix 或 logits，因此新增实验需要补一层统一评估输出。

## 后续新增代码建议

为保持向后兼容，新增功能应尽量以新文件和小型工具模块实现：

- `utils/yelp_data.py`：统一读取 Yelp split、caption、classnames，供 VLM direct、VLM-LoRA 数据构造和诊断脚本复用。
- `utils/eval_outputs.py`：统一保存 `config.yaml`、`predictions.csv`、`metrics.json`、`classification_report.txt`、`confusion_matrix.csv`、`logits.npy`、`parse_failures.csv`。
- `evaluate_vlm_direct.py`：生成式 VLM 直接分类 baseline。
- `build_vlm_lora_dataset.py`：将固定 Yelp split 转为 instruction tuning JSON/JSONL。
- `train_vlm_lora.py`：VLM-LoRA 训练入口和配置模板，独立于现有 CLIP 训练框架。
- `models/tmr_lora.py`：TMR-LoRA expert/routing 模块。
- `train_clip_peft.py`：给 Yelp/CLIP PEFT 提供更清晰的 argparse 入口，同时复用 `main.py`/`Trainer`。
- `diagnose_tmr_lora.py`：class-wise metrics、logit 分布、routing utilization、可选 gradient norm 诊断。
- `tests/`：放 TMR-LoRA 的最小 sanity check，避免影响原有训练脚本。

新增代码不应删除或重写原有入口；需要修改现有 trainer/model 时，优先做可选配置开关，默认行为保持不变。
