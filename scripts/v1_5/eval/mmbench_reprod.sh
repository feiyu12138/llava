#!/bin/bash

SPLIT="mmbench_dev_20230712"
CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
name=llava-v1.5-7b-reprod

# python -m llava.eval.model_vqa_mmbench \
#     --model-path $CKPT \
#     --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$name.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $name
