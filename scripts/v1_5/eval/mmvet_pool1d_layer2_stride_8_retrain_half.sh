#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
layer=2
stride=8
grouping=avgpool1d
name=llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-retrain-half
ckpt=/data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-half
halfpool=True
python -m llava.eval.model_vqa \
    --model-path $ckpt \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images_ \
    --answers-file ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --halfpool $halfpool

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$name.json

