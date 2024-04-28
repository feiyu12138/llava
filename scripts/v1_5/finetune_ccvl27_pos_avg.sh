#!/bin/bash
#
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY='70c34ec6ff006f3a8b19234dd103f67feed8083b'
export WANDB_PROJECT='llava'


layer=0
stride=4
grouping=pos_avg
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/jieneng/data/llava_datasets/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder /data/jieneng/data/llava_datasets/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/luoxin/weights/llava/checkpoint/llava-v1.5-7b-pretrain-stride-4-layer-16-grouping-avgpool1d/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/luoxin/weights/llava/checkpoint/llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping \
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
    --run_name pool4layer1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping 
    #> /data/datasets/jchen293/logs/exp/llava/pool4layer1.log

sleep 2d