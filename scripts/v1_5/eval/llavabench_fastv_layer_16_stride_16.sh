#!/bin/bash

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-fastv-rank-36-k-16
NAME=fastv-rank-36-k-16

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

rank=36
k=16
export OPENAI_API_KEY=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
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank $rank \
    --fast-v-agg-layer $k

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
    --output \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl

python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl
