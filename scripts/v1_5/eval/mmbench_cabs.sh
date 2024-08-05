#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

ROOT_DATA=/data/datasets/data/llava_datasets
ROOT_WEIGHT=/data/datasets/weights/llava/checkpoint

layer=2
stride=4
grouping=cabstractor
name=cabspool4layer2
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping

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
    --grouping $grouping

mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
    --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
    --experiment cabs
