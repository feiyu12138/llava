#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

name=vcc-layer-2-stride-8-fine-3-pool8
grouping=attn
stride=8
layer=2
num_fine_blocks=3
viz_assign=True
selector_type=max_first
savedir=./viz_assign/$name-pope-$selector_type
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/pope/val2014 \
    --answers-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping $grouping \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir \
    --selector_type $selector_type

python llava/eval/eval_pope.py \
    --annotation-dir $ROOT_DATA/eval_luoxin/eval/pope/coco \
    --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$name.jsonl
