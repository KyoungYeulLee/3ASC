#!/bin/bash
# Stage 1 of the GFE embedding model: LoRA pre-training of
# Qwen3-Embedding-8B on (patient symptoms -> OMIM disease text) pairs with an
# InfoNCE objective.
#
#   DATA_ROOT=/path/to/3asc_data bash scripts/embedding/pretrain/train.sh
#
# Merge the LoRA adapter
# afterwards - `swift export --adapters <checkpoint> --merge_lora true` - and
# pass the resulting `<checkpoint>-merged` directory to the finetune script.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/path/to/3asc_data}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}
OUTPUT_DIR=${OUTPUT_DIR:-./output/embedding/pretrain}
nproc_per_node=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=${MASTER_PORT:-24803} \
swift sft \
    --model "$BASE_MODEL" \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type lora \
    --dataset "$DATA_ROOT/embedding/pretraining_dataset.jsonl" \
    --split_dataset_ratio 0.05 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
