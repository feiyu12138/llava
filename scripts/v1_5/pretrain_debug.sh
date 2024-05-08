#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# export NCCL_P2P_DISABLE=1
NNODES=1
GPUS=1
PORT=29600
layer=2
stride=16
grouping=attn
num_fine_blocks=3


ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
LOG_PATH=/data/datasets/jchen293/logs/exp/llava

torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path $ROOT_DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $ROOT_DATA/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $ROOT_WEIGHT/debug \
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
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 
