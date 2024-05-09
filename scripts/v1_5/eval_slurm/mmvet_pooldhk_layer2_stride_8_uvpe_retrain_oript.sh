#!/bin/bash
#
#SBATCH --job-name=mmvet
#SBATCH --error=/datasets/jchen293/logs/exp/llava/1dpool8layer2uvpedhkoript_mmvet.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/1dpool8layer2uvpedhkoript_mmvet.out
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
grouping=detach_hard_k_means
unified_vpe=True

name=llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe-retrain-oript
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe-oript

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

