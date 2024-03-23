#!/bin/bash
#!/bin/bash
#
#SBATCH --job-name=ft_lora_pool8layer16
#SBATCH --error=/datasets/jchen293/logs/exp/llava/ft_lora_pool8layer16.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/ft_lora_pool8layer16.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main

module purge
module load conda
conda activate llava_git


export WANDB_API_KEY='70c34ec6ff006f3a8b19234dd103f67feed8083b'

layer=16
stride=8
grouping=avgpool1d

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /datasets/jchen293/data/llava_datasets/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder /datasets/jchen293/data/llava_datasets/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-pretrain-stride-8-layer-16-grouping-avgpool1d/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-lora-stride-$stride-layer-$layer-grouping-$grouping \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
