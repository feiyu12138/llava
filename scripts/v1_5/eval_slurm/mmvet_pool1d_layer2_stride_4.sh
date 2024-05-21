#!/bin/bash
#
#SBATCH --job-name=1dpool4layer2_mmvet
#SBATCH --error=/datasets/jchen293/logs/exp/llava/1dpool4layer2_mmvet.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/1dpool4layer2_mmvet.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=8

export CUDA_VISIBLE_DEVICES=0

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

layer=2
stride=4
grouping=avgpool1d
unified_vpe=False
name=1dpool4layer2
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-2-grouping-avgpool1d

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

mkdir -p $ROOT_DATA/eval_luoxin/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --dst $ROOT_DATA/eval_luoxin/eval/mm-vet/results/$name.json

