#!/bin/bash
#
#SBATCH --job-name=1dpool4layer2_sqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool4layer2_sqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool4layer2_sqa.out
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
name=1dpool4layer2
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-2-grouping-avgpool1d

python -m llava.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $ROOT_DATA/eval_luoxin/eval/scienceqa/ScienceQA/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping 

python llava/eval/eval_science_qa.py \
    --base-dir $ROOT_DATA/eval_luoxin/eval/scienceqa \
    --result-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --output-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-output.jsonl \
    --output-result $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-result.json
