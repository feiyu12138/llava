#!/bin/bash
# export NCCL_P2P_DISABLE=1
export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_PROJECT='llava_team'
export WANDB_ENTITY='jchen293'

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

NAME=qformer
HASQF=True
NUM_QUERY_TOKEN=32
FREEZEQF=True
QFPATH=$ROOT_WEIGHT/qformer/qformer.bin
QTPATH=$ROOT_WEIGHT/qformer/query_tokens.bin
VMPATH=$ROOT_WEIGHT/qfvm/vm.bin



# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --version plain \
#     --data_path $ROOT_DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder $ROOT_DATA/LLaVA-Pretrain/images \
#     --has_qformer $HASQF \
#     --num_query_token $NUM_QUERY_TOKEN \
#     --freeze_qformer $FREEZEQF \
#     --qformer_path $QFPATH \
#     --vision_model_path $VMPATH \
#     --query_tokens_path $QTPATH \
#     --vision_tower eva_clip_g \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $ROOT_WEIGHT/llava-v1.5-7b-pretrain-$NAME \
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
#     --run_name pt-$NAME


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $ROOT_DATA/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder $ROOT_DATA/LLaVA-Tuning \
    --has_qformer $HASQF \
    --pretrain_mm_mlp_adapter $ROOT_WEIGHT/llava-v1.5-7b-pretrain-$NAME-0520/mm_projector.bin \
    --num_query_token $NUM_QUERY_TOKEN \
    --freeze_qformer $FREEZEQF \
    --qformer_path $QFPATH \
    --vision_model_path $VMPATH \
    --query_tokens_path $QFPATH \
    --vision_tower eva_clip_g \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $ROOT_WEIGHT/llava-v1.5-7b-finetune-$NAME-foreval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ft-$NAME-foreval \
    > /data/datasets/jchen293/logs/exp/llava/qformer/$NAME-foreval.out 2>/data/datasets/jchen293/logs/exp/llava/qformer/$NAME-foreval.err


