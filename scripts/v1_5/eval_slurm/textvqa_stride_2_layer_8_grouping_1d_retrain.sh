#!/bin/bash
#
#SBATCH --job-name=textvqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer21d_textvqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer21d_textvqa.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=main

export CUDA_VISIBLE_DEVICES=0

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

layer=2
stride=8
grouping=avgpool1d
unified_vpe=True
name=llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-retrain
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --unified_vpe $unified_vpe

python -m llava.eval.eval_textvqa \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl
