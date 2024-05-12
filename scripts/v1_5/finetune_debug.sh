#!/bin/bash
NNODES=1
GPUS=1
PORT=29600

DATA_PATH=/data/datasets/jchen293/data/llava_datasets
WEIGHT_PATH=/data/datasets/jchen293/weights/llava/checkpoint

layer=0
stride=1
grouping=pos_avg
rpe=True

torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
     llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder ./playground/data/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $WEIGHT_PATH/llava-v1.5-7b-pretrain-stride-2-layer-16-grouping-avgpool2d/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $WEIGHT_PATH/llava-v1.5-7b-finetune-stride-2-layer-16-grouping-avgpool2d \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --rpe $rpe
