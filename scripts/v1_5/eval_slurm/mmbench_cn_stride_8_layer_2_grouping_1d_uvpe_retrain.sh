#!/bin/bash
#
#SBATCH --job-name=mmbench_cn
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer21duvpe_mmbench_cn.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer21duvpe_mmbench_cn.out
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
name=llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-unifiedvpe-True-retrain
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-8-layer-2-grouping-avgpool1d-unified_vpe-True

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$name.jsonl \
    --lang cn \
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
