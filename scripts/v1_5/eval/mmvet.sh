#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path my-llava-1.5-7b \
    --question-file /home/lye21/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /data/jieneng/data/llava_datasets/eval/mmvet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b.json

