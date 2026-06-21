from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # Directory where datasets are stored
_C.imb_factor = None  # for long-tailed cifar dataset

_C.backbone = ""
_C.resolution = 224

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None  # use manual seed
_C.deterministic = False  # output deterministic results
_C.gpu = None  # assign a single gpu 
_C.num_workers = 20
_C.eval_num_workers = 4  # val/test loaders; lower when running multiple jobs
_C.prefetch_factor = 4
_C.persistent_workers = True
_C.prec = "fp16"  # fp16, fp32, amp

_C.num_epochs = 10
_C.batch_size = 128
_C.micro_batch_size = 128  # for gradient accumulation, must be a divisor of batch size
_C.lr = 0.01
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.optimizer = "Adam"  # "Adam" | "AdamW" | "SGD" — SGD uses momentum
_C.adam_beta1 = 0.9
_C.adam_beta2 = 0.999
_C.adam_eps = 1e-8
_C.loss_type = "LA"  # "CE" / "Focal" / "LDAM" / "CB" / "GRW" / "BS" / "LA" / "LADE"
_C.la_tau = 1.0  # temperature for LogitAdjustedLoss (LA)

_C.classifier = "CosineClassifier"
_C.scale = 25  # for cosine classifier

_C.full_tuning = False  # full fine-tuning
_C.bias_tuning = False  # only fine-tuning the bias 
_C.ln_tuning = False  # only fine-tuning the layer norm
_C.bn_tuning = False  # only fine-tuning the batch norm (only for resnet)
_C.vpt_shallow = False
_C.vpt_deep = False
_C.adapter = False
_C.adaptformer = False
_C.lora = False
_C.lora_mlp = False
_C.tmr_lora = False
_C.tmr_lora_mlp = False
_C.tmr_routing = "uniform"  # "oracle" / "learned" / "uniform" / "random" / "vanilla_lora"
_C.tmr_num_experts = 3
_C.lora_rank = None  # Optional alias for adapter_dim when using LoRA/TMR-LoRA
_C.lora_alpha = 1.0
_C.lora_dropout = 0.0
_C.tmr_head_labels = [1, 2]  # food, inside for Yelp classnames.txt
_C.tmr_tail_labels = [0, 3, 4]  # drink, menu, outside
_C.tmr_oracle_head_weights = [0.5, 0.5, 0.0]
_C.tmr_oracle_tail_weights = [0.5, 0.0, 0.5]
_C.ssf_attn = False
_C.ssf_mlp = False
_C.ssf_ln = False
_C.mask = False   # fine-tuning a specific proportion of all parameters
_C.partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.vpt_len = None  # length of VPT sequence
_C.adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
_C.adaptformer_scale = "learnable"  # "learnable" or scalar
_C.mask_ratio = None
_C.mask_seed = None

_C.init_head = None  # "text_feat" (only for CLIP) / "class_mean" / "1_shot" / "10_shot" / "100_shot" / "linear_probe"
_C.prompt = "default" # "classname" / "default" / "ensemble" / "descriptor"
_C.tte = False  # test-time ensemble
_C.expand = 24 # expand the width and height of images for test-time ensemble
_C.tte_mode = "fivecrop" # "fivecrop" / "tencrop" / "randaug"
_C.randaug_times = 1

_C.zero_shot = False  # zero-shot CLIP (only for CLIP)
_C.text_zeroshot = False  # zero-shot by caption text features (only for CLIP/Yelp-MM)
_C.text_finetune = False  # fine-tune CLIP text encoder for caption classification
_C.test_only = False  # load model and test
_C.test_train = False  # load model and test on the training set
_C.model_dir = None