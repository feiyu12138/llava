#!/bin/bash
#
export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_PROJECT='llava_team'
export WANDB_ENTITY='jchen293'

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

layers=2,2,0
strides=8,2,1
pivots=1730,3460
grouping=avgpool1d
progressive=True
RUN_NAME=1dpool8_2layer2pivot1730_3460_v2


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $ROOT_DATA/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder $ROOT_DATA/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $ROOT_WEIGHT/llava-v1.5-7b-pretrain-reprod/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir  $ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$strides-layer-$layers-grouping-$grouping-progressive-v2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
    --run_name $RUN_NAME \
    --strides $strides \
    --layers $layers \
    --pivots $pivots \
    --grouping $grouping \
    --progressive $progressive \
    1> /data/datasets/jchen293/logs/exp/llava/$grouping-stride-$strides-layer-$layers-progressive-v2.out \
    2> /data/datasets/jchen293/logs/exp/llava/$grouping-stride-$strides-layer-$layers-progressive-v2.err

