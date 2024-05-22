#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

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
savedir=./viz_assign/$name-mmbench_cn
ckpt=/data/jieneng/huggingface/llava-v1.5-7b

python -m llava.eval.model_vqa_mmbench \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$name.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping $grouping \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
    --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
    --experiment $name
