#!/bin/bash
CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
name=llava-v1.5-7b-reprod
# python -m llava.eval.model_vqa_loader \
#     --model-path $CKPT \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/$name.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$name.jsonl
