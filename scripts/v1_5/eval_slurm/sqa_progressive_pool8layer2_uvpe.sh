#!/bin/bash
#
#SBATCH --job-name=sqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer2proguvpe_sqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer2proguvpe_sqa.out
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

python -m llava.eval.model_vqa_science \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $ROOT_DATA/eval_luoxin/eval/scienceqa/ScienceQA/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --unified_vpe $unified_vpe

python llava/eval/eval_science_qa.py \
    --base-dir $ROOT_DATA/eval_luoxin/eval/scienceqa \
    --result-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --output-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-output.jsonl \
    --output-result $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-result.json
