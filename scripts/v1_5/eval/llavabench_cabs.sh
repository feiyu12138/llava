#!/bin/bash
export OPENAI_API_KEY=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

export CUDA_VISIBLE_DEVICES=1

ROOT_DATA=/data/datasets/data/llava_datasets
ROOT_WEIGHT=/data/datasets/weights/llava/checkpoint

layer=2
stride=4
grouping=Convabstractor
name=convabspool4layer2
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping


python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --output \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$name.jsonl

python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$name.jsonl
