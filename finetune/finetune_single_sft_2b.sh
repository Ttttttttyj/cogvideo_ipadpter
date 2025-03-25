#!/bin/bash
CUDA_VISIBLE_DEVICES=5,6
export TRAIN_NUM=100
export MODEL_PATH="./CogVideoX-2b"
export CACHE_PATH="~/.cache"
export DATASET_PATH="/private/task/tuyijing/Datasets/dog_dataset_train" 
export OUTPUT_PATH="./outputs/CogVideoX-2b/ipadapter_4e-5" 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_cogvideox_ipadapter.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --caption_column labels \
  --video_column videos \
  --ref_image_column ref_images \
  --seed 42 \
  --mixed_precision fp16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 8 \
  --num_train_epochs $TRAIN_NUM \
  --checkpointing_steps  400\
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 500 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
