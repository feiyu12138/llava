#!/bin/bash
#
#SBATCH --job-name=pt_p32l16
#SBATCH --error=/datasets/jchen293/logs/exp/llava/%j_0_log.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/%j_0_log.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jchen293@jh.edu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --time=02-00:00:00
#SBATCH --mem=256G

export WANDB_API_KEY='70c34ec6ff006f3a8b19234dd103f67feed8083b'

module purge
module load conda
conda activate llava_git

# which python

# export MASTER_PORT=12800
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr

layer=16
stride=32
grouping=avgpool1d

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /datasets/jchen293/data/llava_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /datasets/jchen293/data/llava_datasets/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-pretrain-bsz16-stride-$stride-layer-$layer-grouping-$grouping \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
    --grouping $grouping


# sbatch --gpus=6000_ada:8 scripts/v1_5/pretrain_slurm_pool16layer16.sh