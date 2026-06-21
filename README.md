# Data Science Practice: AttnRes-LoRA for Yelp Multimodal Long-Tailed Classification

本仓库是中国人民大学《数据科学实践》大作业的实验代码与结果记录。项目围绕 Yelp 餐馆图片五分类任务展开，研究在 **类别长尾** 与 **文本模态缺失** 同时存在的场景下，如何利用本地小规模视觉语言模型（VLM）完成稳定、可复现、可部署的多模态分类。

最终方法是 **AttnRes-LoRA**：一种面向 VLM 的 Cross-Layer Attentive Residual Low-Rank Adaptation 方法。它在冻结 VLM 主干和保留普通 LoRA 更新的基础上，引入跨层视觉/文本摘要记忆，让当前层的 LoRA 残差能够通过 attention 检索历史层信息，从而提升长尾类别和不完整多模态输入下的表现。

论文代码地址：<https://github.com/wuwufufu/Data-Science-Practice>

---

## 1. 项目背景

Yelp 等点评平台包含大量餐馆相关图片和用户 caption。对图片自动分类可以帮助平台组织商户内容，也可以让用户更方便地浏览 food、drink、inside、outside、menu 等不同类型的图片。

这个任务看似是一个普通五分类问题，但实际有两个难点：

1. **类别长尾明显**  
   `food` 是绝对头部类别，而 `menu` 极少。整体统计中，food 占 53.8%，menu 仅占 0.9%，头尾样本数约相差 61 倍。

2. **文本模态不完整**  
   约 51% 样本没有有效 caption。即使有 caption，大部分也只有 2 到 4 个词，文本信息非常短且稀疏。

因此，单纯依赖文本不可行；单纯依赖图像也会在尾部类别和细粒度语义上遇到困难。项目的核心问题是：

> 在长尾类别与文本模态缺失并存的条件下，如何让本地可部署的小 VLM 学会更稳定的多模态分类能力？

---

## 2. 任务定义

主任务是 Yelp 餐馆图片五分类：

| Label | 含义 |
|---|---|
| `food` | 食物图片 |
| `drink` | 饮料图片 |
| `inside` | 餐厅内部环境 |
| `outside` | 餐厅外部环境 |
| `menu` | 菜单、价目表、带文字的菜单图 |

数据划分采用双重分层策略：先按类别分层，再在类别内按 caption 是否缺失分层，尽量保持 train / val / test 中类别分布和文本缺失结构一致。

| Split | drink | food | inside | menu | outside | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1103 | 7524 | 3975 | 122 | 1269 | 13993 |
| Val | 157 | 1074 | 568 | 18 | 181 | 1998 |
| Test | 316 | 2150 | 1137 | 35 | 364 | 4002 |

---

## 3. 实验路线概览

项目不是一开始就直接提出最终方法，而是经历了几个阶段。

### 3.1 描述统计与基础诊断

我们首先分析类别分布、caption 缺失率和 caption 长度分布。结论是：

- 图像是主要可靠信息源；
- 文本在有 caption 时能提供补充，但整体不可依赖；
- `menu` 这类尾部类别需要特别关注；
- 后续方法必须同时处理长尾和多模态缺失。

### 3.2 CLIP 与早期 PEFT Baseline

早期阶段验证了 CLIP zero-shot、image-only / text-only 分类、Adapter、AdaptFormer、LoRA、分类头和 logit-level late fusion。

| 模块 | 关键结果 | 结论 |
|---|---|---|
| image-only vs text-only | image-only 明显优于 text-only | 图像是主模态 |
| CLIP zero-shot | 图像 Acc 约 71.5%，文本 Acc 约 67.4% | 有迁移能力，但不足以解决任务 |
| PEFT 微调 | AdaptFormer 约 96.2 Acc / 93.7 Macro-F1 | 任务适配非常重要 |
| Late fusion | 最优文本权重 alpha 约 0.09 | 文本只应作为小权重辅助 |

### 3.3 本地 VLM 直接测试

我们也测试了直接调用本地 Qwen2.5-VL-3B-Instruct 做分类。Prompt 中是否给出类别解释会影响结果，但 direct VLM 仍明显低于 PEFT 微调后的模型。

| Setting | Accuracy | Macro-F1 | Menu F1 |
|---|---:|---:|---:|
| Image + caption | 85.08 | 77.23 | 68.82 |
| + class definitions | 87.36 | 80.53 | 75.29 |
| + refined definitions | 88.06 | 81.76 | 76.74 |

这一步说明：本地小 VLM 有基础视觉语言能力，但如果不训练，它不能充分适配 Yelp 长尾分布。

### 3.4 普通 VLM-LoRA

普通 LoRA 作为第一个强 baseline，证明 Qwen2.5-VL 可以被有效适配。

| Method | Epoch | Accuracy | Macro-F1 | Menu F1 |
|---|---:|---:|---:|---:|
| VLM-LoRA r8 | 1 | 95.13 | 92.03 | 89.19 |
| VLM-LoRA r8 | 3 | 95.93 | 92.21 | 86.15 |

但普通 LoRA 对所有样本使用同一组低秩更新，面对 food 占比过半、menu 极少的长尾结构时，尾部类仍然不稳定。

### 3.5 多专家 TMRLoRA

我们尝试了“共享 LoRA + 专家 LoRA”的多专家版本。初始动机是：不同类别、不同难度、不同语义类型的样本可能需要不同的参数修正。

| Method | Accuracy | Macro-F1 | Menu F1 | 观察 |
|---|---:|---:|---:|---|
| TMRLoRA r8 1ep | 95.80 | 93.24 | 92.75 | 比普通 LoRA 更均衡 |
| TMRLoRA balanced 1ep | 95.30 | 92.56 | 92.75 | 重采样不一定收益 |
| TMRLoRA r8 3ep | 96.23 | 93.94 | 92.96 | 旧版本最佳 |

多专家确实改善了 macro-F1，但也带来了新的问题：模型必须决定样本应该交给哪个专家。对于只有五个类别的 Yelp 任务，类别专家容易退化成一种硬路由，不够稳定，也不够优雅。

### 3.6 Attention + Expert

随后我们意识到，attention 的本质不是“选一个专家”，而是在候选信息之间做加权汇聚。因此我们尝试把不同层、不同模态的隐藏状态摘要保存下来，让当前 LoRA 更新可以参考历史层的视觉/文本信息。

| Method | Epoch | Accuracy | Macro-F1 | Menu F1 |
|---|---:|---:|---:|---:|
| Depth-TMR r8 | 1 | 95.98 | 93.22 | 91.43 |
| Depth-TMR r8 | 3 | 96.05 | 92.77 | 88.24 |

这个阶段说明跨层 attention 的方向是对的，但如果外面仍然包着专家门控，模型还是会被“分给哪个专家”这个问题牵制。

### 3.7 最终方法：AttnRes-LoRA

最终我们去掉门控和类别专家，只保留注意力式残差记忆。

核心形式可以写成：

```text
y = W0 x + BAx + lambda * Attn(q, K_mem, V_mem)
```

其中：

- `W0 x` 是冻结 VLM 主干的原始输出；
- `BAx` 是标准 LoRA 的低秩残差；
- `Attn(q, K_mem, V_mem)` 是从历史 block 的 multimodal summaries 中检索得到的跨层上下文；
- `lambda` 控制 attention residual 的注入强度。

这版没有类别专家、没有 MoE router、没有路由辅助损失。它把问题从“样本应该分给哪个专家”改成“当前层应该从历史层检索哪些视觉/文本信息”。

最终结果：

| Method | Accuracy | Macro-F1 | Menu F1 |
|---|---:|---:|---:|
| AttnRes-LoRA | **97.38** | **94.45** | **95.21** |

---

## 4. 与 LoRA 改进方法的比较

为了说明最终方法不是简单调参，我们比较了多种 LoRA-family baseline。

| Method | Accuracy / Macro-F1 | 核心思想 |
|---|---:|---|
| LoRA | 95.93 / 92.21 | 标准低秩适配 |
| AdaLoRA | 95.70 / 92.05 | 动态分配 rank 预算 |
| DoRA | 96.18 / 92.68 | 权重幅值/方向分解 |
| LoRA+ | 96.05 / 92.45 | A/B 矩阵使用不同学习率 |
| rsLoRA | 96.00 / 92.36 | alpha/r 改为 alpha/sqrt(r) |
| MixLoRA | 96.42 / 93.02 | LoRA experts + top-k router |
| MiLoRA | 95.55 / 91.92 | 利用 minor singular components |
| HydraLoRA | 95.82 / 92.18 | 非对称 LoRA 结构 |
| MoA | 96.55 / 93.18 | heterogeneous mixture of adapters |
| AttnRes-LoRA | **97.38 / 94.45** | 跨层 attention residual 记忆 |

这些 baseline 分别代表 rank 分配、权重分解、优化动力学、缩放策略、专家混合和结构增强等方向。AttnRes-LoRA 的不同点在于：它不是只增强当前层 LoRA 的表达能力，而是让 LoRA 残差能够显式检索历史层的多模态摘要。

---

## 5. 跨 Backbone 与跨数据集验证

我们还在多个 2B 到 3B 级别的本地 VLM 上测试了 AttnRes-LoRA 的可迁移性。

| Backbone | Accuracy | Macro-F1 | 结果 |
|---|---:|---:|---|
| Qwen3-VL-2B | 97.12 | 94.21 | 稳定有效 |
| InternVL3-2B | 96.86 | 93.88 | 稳定有效 |
| SmolVLM2-2.2B | 96.18 | 93.06 | 有效 |
| PaliGemma 2-3B | 96.74 | 93.71 | 有效 |

跨数据集结果覆盖多标签图文分类、多模态推理/VQA、OCR/文字图像理解和图文冲突识别。

| Dataset | Task | Result |
|---|---|---|
| MM-IMDb / MM-IMDb2 | 多标签图文分类 | 85.2 / 83.9; 83.4 / 81.7 |
| GQA / ScienceQA | 多模态推理 / VQA | 63.8 Acc; 92.4 Acc |
| TextVQA | OCR / 文字图像理解 | 80.8 Acc |
| Hateful Memes | 图文冲突 / 模态互补 | 88.1 AUROC / 79.6 Acc |

---

## 6. 代码结构

仓库主要代码位于 `LIFT-Yelp-code/`。

```text
LIFT-Yelp-code/
├── evaluate_vlm_direct.py        # 直接调用本地 VLM 做 Yelp 分类
├── build_vlm_lora_dataset.py     # 构造 VLM LoRA 训练数据
├── train_vlm_lora.py             # VLM-LoRA / TMRLoRA / Depth-TMR / AttnRes-LoRA 主入口
├── vlm_tmr_lora.py               # VLM 侧 TMR、Depth-TMR、AttnRes-LoRA 核心实现
├── train_clip_peft.py            # 早期 CLIP/PEFT 实验入口
├── diagnose_tmr_lora.py          # TMRLoRA 诊断脚本
├── datasets/                     # 数据集 split 与 dataset loader
├── models/                       # CLIP 侧 PEFT / TMR 模块
├── utils/                        # 配置、评估、采样、Yelp 数据工具
├── configs/                      # 原始配置文件
├── scripts/                      # 批量实验脚本
├── outputs/                      # 轻量实验记录：config、metrics、logs、loss 曲线
└── docs/                         # 项目说明文档
```

几个最关键的文件：

| 文件 | 作用 |
|---|---|
| `train_vlm_lora.py` | 主训练入口，支持普通 LoRA、TMRLoRA、Depth-TMR、AttnRes-LoRA |
| `vlm_tmr_lora.py` | 最终方法和中间方法的核心 adapter 实现 |
| `evaluate_vlm_direct.py` | 本地 VLM direct prompting baseline |
| `build_vlm_lora_dataset.py` | 从图像路径、caption、label 构造训练样本 |
| `utils/eval_outputs.py` | 指标计算、分类报告、预测输出 |
| `datasets/yelp_lt.py` | Yelp 数据加载逻辑 |

---

## 7. 环境与依赖

原始代码基于 Python / PyTorch / HuggingFace Transformers / PEFT 生态。建议环境包括：

```bash
pip install torch transformers accelerate peft safetensors qwen-vl-utils scikit-learn pillow tqdm pyyaml
```

如果使用 Qwen2.5-VL / Qwen3-VL，需要保证对应模型权重已下载到本地，并且 transformers 版本支持相应模型结构。

本项目中的大模型权重和原始图片数据没有放进代码包，需要单独准备。

---

## 8. 复现入口

进入代码目录：

```bash
cd LIFT-Yelp-code
```

查看 direct VLM baseline 参数：

```bash
python3 evaluate_vlm_direct.py --help
```

构造 VLM LoRA 数据：

```bash
python3 build_vlm_lora_dataset.py --help
```

训练或评估 LoRA / TMRLoRA / AttnRes-LoRA：

```bash
python3 train_vlm_lora.py --help
```

由于不同机器上的模型路径、数据路径和显存条件不同，推荐先参考 `outputs/*/config.yaml` 中记录的实验配置，再调整本地路径。

---

## 9. 打包说明

当前代码包保留了完整源码和轻量实验记录，但没有包含以下大文件：

- VLM backbone 权重；
- Yelp 原始图片压缩包；
- 训练得到的 adapter 权重：`*.pt`、`*.pth`、`*.bin`、`*.safetensors`；
- optimizer / scheduler 状态；
- Python 缓存文件。

这些文件不是源码本身，而且体积较大。若需要完整复现实验，需要在本地重新下载模型和数据，并用仓库中的脚本重新训练或评估。

---

## 10. 项目结论

本项目的主要结论是：

1. Yelp 五分类不是普通均衡分类任务，而是长尾 + 文本缺失的多模态任务。  
2. 直接调用本地小 VLM 有一定能力，但远不足以解决任务。  
3. 普通 LoRA 能显著提升性能，但尾部类仍不稳定。  
4. 多专家 TMRLoRA 提升 macro-F1，但专家门控会引入新的不稳定性。  
5. 将 attention 作为跨层信息加权机制是有效方向。  
6. 最终的 AttnRes-LoRA 去掉专家和门控，只保留跨层 attention residual，在 Yelp 上达到 **97.38 Accuracy / 94.45 Macro-F1**，并在多个本地小 VLM 和其他多模态任务上保持有效。

一句话概括：

> AttnRes-LoRA 不是让样本选择专家，而是让当前层从历史视觉/文本记忆中按需检索信息。

---

## 11. Authors

- Haoran Zhang, School of Statistics, Renmin University of China
- Zhiyuan Huang, School of Statistics, Renmin University of China
- Xingshuo Zhang, School of Statistics, Renmin University of China
- Yuqian Zhou, School of Applied Economics, Renmin University of China
- Shouchen Shi, School of Applied Economics, Renmin University of China
