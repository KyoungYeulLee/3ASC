#!/bin/bash
# Stage 2 of the GFE embedding model: LoRA fine-tuning of the
# merged pre-trained model on (patient symptoms -> in-house disease record)
# pairs with an InfoNCE objective.
#
#   PRETRAIN_MERGED=./output/embedding/pretrain/<version>/checkpoint-<n>-merged \
#   DATA_ROOT=/path/to/3asc_data \
#   bash scripts/embedding/finetune/train.sh
#
# Merge the LoRA adapter
# afterwards - `swift export --adapters <checkpoint> --merge_lora true` - and
# use the resulting `<checkpoint>-merged` directory to extract the embeddings
# the MIL-GFE model consumes.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/path/to/3asc_data}
PRETRAIN_MERGED=${PRETRAIN_MERGED:?set PRETRAIN_MERGED to the merged pre-trained model directory}
OUTPUT_DIR=${OUTPUT_DIR:-./output/embedding/finetune}
nproc_per_node=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=${MASTER_PORT:-24740} \
swift sft \
    --model "$PRETRAIN_MERGED" \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type lora \
    --dataset "$DATA_ROOT/embedding/finetuning_dataset.jsonl" \
    --split_dataset_ratio 0.05 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
