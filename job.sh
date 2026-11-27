#!/usr/bin/env bash
set -euo pipefail
cd "$1"
work=$2
py=${CS290_PYTHON:-"$PWD/.venv/bin/python"}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
trap 'echo "Pilot stopped at line $LINENO. Completed outputs and logs are retained." >&2' ERR

echo "Slurm job ${SLURM_JOB_ID:?} on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$py" pilot.py cuda --work "$work"
"$py" pilot.py preflight --work "$work"

lora="$work/rain_lora"
if [[ ! -s "$lora/pytorch_lora_weights.safetensors" ]]; then
    resume=()
    if compgen -G "$lora/checkpoint-*" >/dev/null; then
        resume=(--resume_from_checkpoint latest)
    fi
    "$py" vendor/diffusers/train_text_to_image_lora.py \
        --pretrained_model_name_or_path "$work/models/sd15" \
        --train_data_dir "$work/data/rain_train" --output_dir "$lora" \
        --resolution 512 --train_batch_size 1 --gradient_accumulation_steps 1 \
        --rank 8 --max_train_steps "${CS290_TRAIN_STEPS:-600}" \
        --learning_rate 0.0001 --lr_scheduler constant --lr_warmup_steps 0 \
        --mixed_precision fp16 --allow_tf32 --gradient_checkpointing \
        --checkpointing_steps 200 --checkpoints_total_limit 2 --seed 290 "${resume[@]}"
fi
test -s "$lora/pytorch_lora_weights.safetensors"
"$py" pilot.py baseline --work "$work"
"$py" pilot.py sms --work "$work" --steps "${CS290_SMS_STEPS:-500}"
"$py" pilot.py report --work "$work"
