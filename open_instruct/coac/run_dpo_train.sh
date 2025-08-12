#!/bin/bash
export PYTHONPATH="/home/ving/tulu:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1

NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  /home/ving/tulu/open_instruct/dpo_tune_cache.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name meta-llama/Llama-3.2-1B-Instruct \
  --use_flash_attn False \
  --gradient_checkpointing \
  --use_slow_tokenizer \
  --use_lora True \
  --merge_lora False \
  --dataset_mixer_list /home/ving/tulu/open_instruct/coac/data/pharma_dpo_fixed.jsonl 1.0 \
  --max_seq_length 2048 \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 5e-7 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --weight_decay 0.0 \
  --num_train_epochs 3 \
  --output_dir /home/ving/tulu/open_instruct/coac/output/dpo_llama_1b \
  --with_tracking False \
  --report_to none \
  --push_to_hub False \
  --logging_steps 1
