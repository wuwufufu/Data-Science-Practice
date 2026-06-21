#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash train_peft_modules.sh
#   bash train_peft_modules.sh --gpus 0,1 --parallel-jobs 2
#   bash train_peft_modules.sh --modules adaptformer,adapter,lora --gpus 2

dataset="yelp_lt"
model="clip_vit_b16"
loss_type="CE"
classifier="CosineClassifier"

# Keep base settings unchanged.
base_num_workers=16
base_eval_num_workers=4
prefetch_factor=4
persistent_workers=True

gpus_csv="2"
parallel_jobs=1
modules_csv="adaptformer,adapter,lora,lora_mlp,vpt_shallow,vpt_deep,ssf_attn,ssf_mlp,ssf_ln,ln_tuning,bias_tuning,full_tuning"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      gpus_csv="$2"
      shift 2
      ;;
    --parallel-jobs)
      parallel_jobs="$2"
      shift 2
      ;;
    --modules)
      modules_csv="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

IFS=',' read -r -a gpus <<< "$gpus_csv"
IFS=',' read -r -a modules <<< "$modules_csv"

if [[ ${#gpus[@]} -eq 0 ]]; then
  echo "No GPUs provided."
  exit 1
fi

if (( parallel_jobs < 1 )); then
  echo "--parallel-jobs must be >= 1"
  exit 1
fi

# Reduce loader pressure when running multiple trainings in parallel.
num_workers=$(( base_num_workers / parallel_jobs ))
eval_num_workers=$(( base_eval_num_workers / parallel_jobs ))
if (( num_workers < 2 )); then num_workers=2; fi
if (( eval_num_workers < 1 )); then eval_num_workers=1; fi

all_modules=(
  full_tuning
  bias_tuning
  ln_tuning
  vpt_shallow
  vpt_deep
  adapter
  adaptformer
  lora
  lora_mlp
  ssf_attn
  ssf_mlp
  ssf_ln
)

build_common_opts() {
  local gpu="$1"
  local module="$2"
  local run_tag="$3"
  local short_output_dir="peft_${dataset}_${model}_${module}_gpu${gpu}_${run_tag}"
  cat <<EOF
-d ${dataset} -m ${model} \
gpu ${gpu} \
loss_type ${loss_type} \
classifier ${classifier} \
output_dir ${short_output_dir} \
num_workers ${num_workers} \
eval_num_workers ${eval_num_workers} \
prefetch_factor ${prefetch_factor} \
persistent_workers ${persistent_workers}
EOF
}

build_module_opts() {
  local target_module="$1"
  local opts=""
  local module
  for module in "${all_modules[@]}"; do
    opts+=" ${module} False"
  done
  opts+=" ${target_module} True"
  if [[ "${target_module}" == "vpt_shallow" || "${target_module}" == "vpt_deep" ]]; then
    opts+=" vpt_len 10"
  fi
  echo "${opts}"
}

extract_epoch1_speed() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys

log_file = sys.argv[1]
pat = re.compile(
    r"epoch \[1/10\] batch \[\d+/\d+\].*?time [0-9.]+ \(([0-9.]+)\).*?data [0-9.]+ \(([0-9.]+)\)"
)
time_avg = None
data_avg = None
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            time_avg = m.group(1)
            data_avg = m.group(2)

if time_avg is None or data_avg is None:
    print("not_ready")
else:
    print(f"data_avg={data_avg}  time_avg={time_avg}")
PY
}

echo "Running modules: ${modules_csv}"
echo "GPUs: ${gpus_csv}"
echo "parallel_jobs=${parallel_jobs}, num_workers=${num_workers}, eval_num_workers=${eval_num_workers}"

running=0
gpu_idx=0
run_tag="$(date +%m%d_%H%M%S)"
declare -a launched_logs=()

for module in "${modules[@]}"; do
  gpu="${gpus[$(( gpu_idx % ${#gpus[@]} ))]}"
  gpu_idx=$((gpu_idx + 1))

  log_file="train_${dataset}_${model}_${module}_${loss_type}_${classifier}_gpu${gpu}.log"
  launched_logs+=("${log_file}")

  common_opts="$(build_common_opts "${gpu}" "${module}" "${run_tag}")"
  module_opts="$(build_module_opts "${module}")"

  # shellcheck disable=SC2086
  nohup python main.py ${common_opts} ${module_opts} > "${log_file}" 2>&1 &
  echo "[LAUNCH] module=${module} gpu=${gpu} log=${log_file} pid=$!"

  running=$((running + 1))
  if (( running >= parallel_jobs )); then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All runs finished."
echo
echo "==== Epoch-1 speed summary (for parallel slowdown check) ===="
for log_file in "${launched_logs[@]}"; do
  echo "${log_file}: $(extract_epoch1_speed "${log_file}")"
done
