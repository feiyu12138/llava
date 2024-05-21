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
    --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --has_qformer $HASQF \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$CKPT.jsonl
