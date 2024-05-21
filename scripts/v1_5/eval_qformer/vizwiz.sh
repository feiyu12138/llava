#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

CKPT=llava-v1.5-7b-finetune-qformer
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"
HASQF=True

python -m llava.eval.model_vqa_loader \
    --model-path $ROOT_WEIGHT/$CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/vizwiz/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --has_qformer $HASQF \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers_upload/$CKPT.json
