#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

grouping=none
stride=8
layer=2
unified_vpe=False
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-avgpool1d-unified_vpe-$unified_vpe-progressive
name=llava-v1.5-7b-progressive

# python -m llava.eval.model_vqa_loader \
#     --model-path $ckpt \
#     --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
#     --image-folder $ROOT_DATA/eval_luoxin/eval/pope/val2014 \
#     --answers-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir $ROOT_DATA/eval_luoxin/eval/pope/coco \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl
