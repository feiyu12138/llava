#!/bin/bash
#
#SBATCH --job-name=1dpool64layer2retrain_mmvet
#SBATCH --error=/datasets/jchen293/logs/exp/llava/1dpool64layer2retrain_mmvet.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/1dpool64layer2retrain_mmvet.out
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
stride=64
grouping=avgpool1d
unified_vpe=False
name=1dpool64layer2retrain
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-2-grouping-avgpool1d

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/mm-vet/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --unified_vpe $unified_vpe

mkdir -p $ROOT_DATA/eval_luoxin/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --dst $ROOT_DATA/eval_luoxin/eval/mm-vet/results/$name.json

