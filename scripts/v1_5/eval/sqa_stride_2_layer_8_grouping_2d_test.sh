#!/bin/bash
layer=1
stride=2
grouping=avgpool2d
export CUDA_VISIBLE_DEVICES=7
name=stride-$stride-layer-$layer-grouping-$grouping
CKPT="/home/jchen293/llava/checkpoints/llava-v1.5-7b-$name"
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /data/jieneng/data/llava_datasets/eval/scienceqa/ScienceQA/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping 

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers/$name-output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers/$name-result.json
