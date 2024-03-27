#!/bin/bash
#
#SBATCH --job-name=pt_MSAgate4layer8_k7_rps
#SBATCH --error=/datasets/jchen293/logs/exp/llava/pt_MSAgate4layer8_k7_rps.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/pt_MSAgate4layer8_k7_rps.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --exclude=ccvl[14,33-38]

export WANDB_API_KEY='70c34ec6ff006f3a8b19234dd103f67feed8083b'
export WANDB_PROJECT='llava'

module purge
module load conda
conda activate llava_git

layer=16
stride=4
grouping=DWConvabstractor_gate
abstractor_kernel_size=7
abstractor_rel_pos_spatial=True
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /datasets/jchen293/data/llava_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /datasets/jchen293/data/llava_datasets/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_abstractor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-pretrain-stride-$stride-layer-$layer-grouping-$grouping \
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
    --run_name pt_MSAgate4layer8_k7_rps \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --abstractor_kernel_size $abstractor_kernel_size \
    --abstractor_rel_pos_spatial $abstractor_rel_pos_spatial
