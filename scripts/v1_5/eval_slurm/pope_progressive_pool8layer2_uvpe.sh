#!/bin/bash
#
#SBATCH --job-name=pope
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer2proguvpe_pope.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer2proguvpe_pope.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=main

export CUDA_VISIBLE_DEVICES=0

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

unified_vpe=True
stride=8
layer=2
grouping=avgpool1d
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-8-layer-2-grouping-avgpool1d-unified_vpe-True-progressive
name=1dpool8layer2proguvpe

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/pope/val2014 \
    --answers-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --unified_vpe $unified_vpe

python llava/eval/eval_pope.py \
    --annotation-dir $ROOT_DATA/eval_luoxin/eval/pope/coco \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl
