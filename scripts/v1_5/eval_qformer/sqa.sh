#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

CKPT=llava-v1.5-7b-finetune-qformer
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"
HASQF=True

python -m llava.eval.model_vqa_science \
    --model-path $ROOT_WEIGHT/$CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $ROOT_DATA/eval_luoxin/eval/scienceqa/images/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --has_qformer $HASQF \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir $ROOT_DATA/eval_luoxin/eval/scienceqa \
    --result-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$CKPT-output.jsonl \
    --output-result $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$CKPT-result.json
