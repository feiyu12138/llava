#!/bin/bash
grouping=avgpool2d
for stride in 2 4 8; do
    for layer in 1 8 16 30; do
    python -m llava.eval.model_vqa_science \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder /data/jieneng/data/llava_datasets/eval/scienceqa/ScienceQA/test \
        --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-layer-$layer-stride-$stride.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --stride $stride \
        --layer $layer \
        --grouping $grouping

    python llava/eval/eval_science_qa.py \
        --base-dir ./playground/data/eval/scienceqa \
        --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-layer-$layer-stride-$stride.jsonl \
        --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_output-layer-$layer-stride-$stride.jsonl \
        --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_result-layer-$layer-stride-$stride.json
    done
done
