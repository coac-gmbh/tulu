export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH=/home/projects2/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
OUTPUT_PATH=/home/projects2/dagr/models/llama3-freiraum
DATA_PATH=/home/dagr/general-extraction/llama_finetune/data/freiraum
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${MODEL_PATH} \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name ${MODEL_PATH} \
    --use_slow_tokenizer \
    --dataset_mixer_list ${DATA_PATH} 1.0 \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 1000 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir ${OUTPUT_PATH}/lora/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path ${MODEL_PATH} \
    --lora_model_name_or_path ${OUTPUT_PATH}/lora/ \
    --output_dir ${OUTPUT_PATH} \
    --save_tokenizer
