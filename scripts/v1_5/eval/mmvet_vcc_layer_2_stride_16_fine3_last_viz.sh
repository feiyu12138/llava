#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name=vcc-layer-2-stride-16-fine-3
grouping=attn
stride=16
layer=2
num_fine_blocks=3
viz_assign=True
savedir=./viz_assign/$name-mmvet
ckpt=/home/lye21/llava_git/llava/playground/data/checkpoint/llava-v1.5-7b-stride-16-layer-2-grouping-attn-num_fine_block-3
python -m llava.eval.model_vqa \
    --model-path $ckpt \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping $grouping \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$name.json

