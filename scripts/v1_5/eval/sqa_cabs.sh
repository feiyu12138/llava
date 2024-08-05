#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

ROOT_DATA=/data/datasets/data/llava_datasets
ROOT_WEIGHT=/data/datasets/weights/llava/checkpoint

layer=2
stride=4
grouping=cabstractor
name=cabspool4layer2
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file $ROOT_DATA/eval_luoxin/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $ROOT_DATA/eval_luoxin/eval/scienceqa/ScienceQA/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir $ROOT_DATA/eval_luoxin/eval/scienceqa \
    --result-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --output-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-output.jsonl \
    --output-result $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-result.json
