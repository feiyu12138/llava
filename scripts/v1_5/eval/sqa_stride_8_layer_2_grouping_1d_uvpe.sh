#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
layer=2
stride=8
grouping=avgpool1d
name=llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-uvpe
unified_vpe=True

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

# python -m llava.eval.model_vqa_science \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder /data/jieneng/data/llava_datasets/eval/scienceqa/ScienceQA/test \
#     --answers-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --layer $layer \
#     --stride $stride \
#     --grouping $grouping \
#     --unified_vpe $unified_vpe

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$name-result.json
