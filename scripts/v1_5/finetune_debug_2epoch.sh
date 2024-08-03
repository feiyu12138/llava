#!/bin/bash

#SBATCH --job-name=debug_2epoch
#SBATCH --error=/datasets/jchen293/logs/exp/gen_llava/debug_2epoch.err
#SBATCH --output=/datasets/jchen293/logs/exp/gen_llava/debug_2epoch.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=80

export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_PROJECT='llava_team'
export WANDB_ENTITY='jchen293'

module purge
module load conda
conda activate llava_git

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /datasets/jchen293/data/llava_datasets/LLaVA-Tuning/llava_v1_5_mix665k_ocr_filtered.json \
    --image_folder /datasets/jchen293/data/llava_datasets/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-2epoch-debug \
    --num_train_epochs 2 \
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
    --run_name debug_2epoch \
    > /datasets/jchen293/logs/exp/gen_llava/llava_2epoch_debug.log

sleep 2d
