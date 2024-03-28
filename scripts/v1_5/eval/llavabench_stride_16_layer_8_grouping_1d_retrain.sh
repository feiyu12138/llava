#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
layer=8
stride=16
grouping=avgpool1d
name=stride-$stride-layer-$layer-grouping-$grouping
CKPT="/home/jchen293/llava/checkpoints/llava-v1.5-7b-$name"
export OPENAI_API_KEY=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
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context  ./playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
         ./playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl > playground/data/eval/llava-bench-in-the-wild/review_result/$name-train.txt
