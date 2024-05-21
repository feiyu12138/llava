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
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --has_qformer $HASQF \
    --conv-mode vicuna_v1

cd $ROOT_DATA/eval_luoxin/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
