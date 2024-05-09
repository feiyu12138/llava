#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"
grouping=none
stride=8
layer=2
unified_vpe=False
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-avgpool1d-unified_vpe-$unified_vpe-progressive
name=llava-v1.5-7b-progressive

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd $ROOT_DATA/eval_luoxin/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
