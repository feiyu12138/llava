#!/bin/bash
#
#SBATCH --job-name=pope
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/pool8layer0prog2600_pope.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/pool8layer0prog2600_pope.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=main

export CUDA_VISIBLE_DEVICES=0

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

unified_vpe=False
stride=8
layer=0
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-avgpool1d-unified_vpe-$unified_vpe-progressive
name=pool8layer0prog2600

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/pope/val2014 \
    --answers-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir $ROOT_DATA/eval_luoxin/eval/pope/coco \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl
