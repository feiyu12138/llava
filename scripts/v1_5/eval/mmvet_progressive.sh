#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

name=pool1d-progressive
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"
grouping=none
stride=8
layer=2
unified_vpe=False
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-avgpool1d-unified_vpe-$unified_vpe-progressive

python -m llava.eval.model_vqa \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/mm-vet/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping 

mkdir -p $ROOT_DATA/eval_luoxin/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$name.jsonl \
    --dst $ROOT_DATA/eval_luoxin/eval/mm-vet/results/$name.json

