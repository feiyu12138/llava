#!/bin/bash


CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
name=llava-v1.5-7b-reprod


python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question /data/luoxin/data/llava/llava-bench-in-the-wild/questions.jsonl \
    --context /data/luoxin/data/llava/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        /data/luoxin/data/llava/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl
