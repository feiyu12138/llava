#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
NNODES=1
GPUS=1
PORT=29600
layer=2
stride=8
grouping=attn
num_fine_blocks=2
name=llava-7b-v1.5-layer-$layer-stride-$stride-grouping-$grouping-num_fine_blocks-$num_fine_blocks
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder ./playground/data/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$name \
    --group_by_modality_length True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 
