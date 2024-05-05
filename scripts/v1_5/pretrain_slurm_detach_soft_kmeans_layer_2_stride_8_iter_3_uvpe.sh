#!/bin/bash
#
#SBATCH --job-name=pt_dsk_l2s8i3_uvpe
#SBATCH --error=/datasets/jchen293/logs/exp/llava/pt_dsk_l2s8i3_uvpe.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/pt_dsk_l2s8i3_uvpe.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main

export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_PROJECT='llava_team'

module purge
module load conda
conda activate llava_git

layer=2
stride=8
halfpool=False
grouping=detach_soft_k_means
unified_vpe=True
citer=3

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path $ROOT_DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $ROOT_DATA/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_abstractor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $ROOT_WEIGHT/llava-v1.5-7b-pretrain-stride-$stride-layer-$layer-grouping-$grouping-citer-$citer \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
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
    --run_name ptdhkmeanslayer2stride8 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --halfpool $halfpool \
    --unified_vpe $unified_vpe \
    --citer $citer
