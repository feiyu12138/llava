#!/bin/bash
#
# export NCCL_P2P_DISABLE=1
export WANDB_API_KEY='70c34ec6ff006f3a8b19234dd103f67feed8083b'
export WANDB_PROJECT='llava_team'

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

layer=2
stride=8
grouping=avgpool1d
unified_vpe=True


# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --version plain \
#     --data_path /data/datasets/jchen293/data/llava_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder /data/datasets/jchen293/data/llava_datasets/LLaVA-Pretrain/images \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir  /data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-pretrain-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --stride $stride \
#     --layer $layer \
#     --grouping $grouping \
#     --unified_vpe $unified_vpe \
#     > /data/datasets/jchen293/logs/exp/llava/pt_$grouping-stride-$stride-layer-$layer-uvpe-$unified_vpe.log


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/datasets/jchen293/data/llava_datasets/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder /data/datasets/jchen293/data/llava_datasets/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-pretrain-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir  /data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe \
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
    --run_name pool8layer21duvpe_debug \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --unified_vpe $unified_vpe \
    1> /data/datasets/jchen293/logs/exp/llava/$grouping-stride-$stride-layer-$layer-uvpe-$unified_vpe.out \
    2> /data/datasets/jchen293/logs/exp/llava/$grouping-stride-$stride-layer-$layer-uvpe-$unified_vpe.err

sleep 2d
