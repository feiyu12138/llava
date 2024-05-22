#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

name=vcc-layer-2-stride-16-fine-3-wotrain-jn
grouping=attn
stride=16
layer=2
num_fine_blocks=3
viz_assign=True
savedir=./viz_assign/$name-textvqa
ckpt=/data/jieneng/huggingface/llava-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping $grouping \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir

python -m llava.eval.eval_textvqa \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl
