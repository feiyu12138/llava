#!/bin/bash
layer=16
stride=2
grouping=none
CKPT="/home/lye21/LLaVA/checkpoints/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool2d"
name=llava-v1.5-7b-stride-4-layer-16-grouping-avgpool2d
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping 

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$name.json

