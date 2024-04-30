#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name=llava-v1.5-7b-vcc-layer-2-stride-32-fine-2-avg
grouping=attn
stride=32
layer=2
num_fine_blocks=2
python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images_ \
    --answers-file ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping $grouping \
    --stride $stride \
    --layer $layer \
    --num-fine-blocks $num_fine_blocks \
    --explore-prob 0.0 \

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$name.json

