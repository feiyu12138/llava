#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,3
export NCCL_P2P_DISABLE=1
NNODES=1
GPUS=3
PORT=29600
rank=72
k=2
use_fast_v=True
fast_v_sys_length=36
fast_v_image_token_length=576
name=llava-v1.5-7b-fastv-rank-$rank-k-$k
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$name \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
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
    --use_fast_v $use_fast_v \
    --fast_v_sys_length $fast_v_sys_length \
    --fast_v_image_token_length $fast_v_image_token_length \
    --fast_v_attention_rank $rank \
    --fast_v_agg_layer $k 
