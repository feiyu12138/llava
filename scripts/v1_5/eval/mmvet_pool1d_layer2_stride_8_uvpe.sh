#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
layer=2
stride=8
grouping=avgpool1d
name=llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-uvpe-new
unified_vpe=True

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file $ROOT_DATA/eval_luoxin/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/mm-vet/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --unified_vpe $unified_vpe

mkdir -p $ROOT_DATA/eval_luoxin/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --dst $ROOT_DATA/eval_luoxin/eval/mm-vet/results/$name.json

