#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

CKPT=llava-v1.5-7b-finetune-qformer
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"
HASQF=True

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $ROOT_WEIGHT/$CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --has_qformer $HASQF \
    --conv-mode vicuna_v1

mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
    --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
