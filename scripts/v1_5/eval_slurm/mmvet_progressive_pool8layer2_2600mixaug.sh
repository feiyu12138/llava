#!/bin/bash
#
#SBATCH --job-name=1dpool8layer2progmixaug_mmvet
#SBATCH --error=/datasets/jchen293/logs/exp/llava/1dpool8layer2progmixaug_mmvet.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/1dpool8layer2progmixaug_mmvet.out
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
layer=2
grouping=avgpool1d
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8layer2progmixaug
name=1dpool8layer2progmixaug

python -m llava.eval.model_vqa \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/mm-vet/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $ROOT_DATA/eval_luoxin/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --dst $ROOT_DATA/eval_luoxin/eval/mm-vet/results/$name.json

