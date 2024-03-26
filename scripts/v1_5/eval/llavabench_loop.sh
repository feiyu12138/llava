#!/bin/bash
grouping=avgpool2d
for stride in 2 4 8; do
    for layer in 1 8 16 30; do
python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b-$stride-layer-$layer.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping
done
done

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question /data/luoxin/data/llava/llava-bench-in-the-wild/questions.jsonl \
    --context /data/luoxin/data/llava/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        /data/luoxin/data/llava/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b-$stride-layer-$layer.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b-$stride-layer-$layer.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b-stride-$stride-layer-$layer.jsonl
