#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-reprod-v4

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

layer=4,12,24
stride=2
grouping=avgpool1d


NAME=multi-level-stride-2-infer_new
python -m llava.eval.model_vqa_loader_flops \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping avgpool1d

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
