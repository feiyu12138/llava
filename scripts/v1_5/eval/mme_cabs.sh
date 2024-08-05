#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

ROOT_DATA=/data/datasets/data/llava_datasets
ROOT_WEIGHT=/data/datasets/weights/llava/checkpoint

layer=2
stride=4
grouping=cabstractor
name=cabspool4layer2
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \

cd $ROOT_DATA/eval_luoxin/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name > ./eval_result/$name.txt
