#!/bin/bash

# ===============================
# GRPO Training: Qwen2.5-VL + CLEVR + DeepSpeed
# ===============================

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_clevr_ds.txt"

QWEN_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
HF_DATASET="leonardPKU/clevr_cogen_a_train"
OUTPUT_DIR="outputs/qwen2.5vl-clevr-grpo-ds"
RUN_NAME="Qwen2.5VL-GRPO-CLEVR-Deepspeed"
DS_CONFIG="local_scripts/zero1_no_optimizer.json"

# Create output dir if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Launch training
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  src/open_r1/grpo.py \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${QWEN_PATH} \
  --dataset_name ${HF_DATASET} \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --min_pixels 3136 \
  --max_pixels 501760 \
  --num_train_epochs 2 \
  --run_name ${RUN_NAME} \
  --save_steps 200 \
  --save_total_limit 3 \
  --save_only_model true \
  --report_to wandb \
  --temperature 1.0 \
  --num_generations 2 \
  --deepspeed ${DS_CONFIG} \
  2>&1 | tee "${OUTPUT_DIR}/training_log.txt"

