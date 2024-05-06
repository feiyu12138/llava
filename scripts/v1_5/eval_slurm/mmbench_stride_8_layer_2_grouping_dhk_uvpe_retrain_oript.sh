#!/bin/bash
#
#SBATCH --job-name=mmbench
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/dhkpool8layer2uvpeoript_mmbench.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/dhkpool8layer2uvpeoript_mmbench.out
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
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --unified_vpe $unified_vpe

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
    --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
    --experiment $name
