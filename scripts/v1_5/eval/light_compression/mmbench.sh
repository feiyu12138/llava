#!/bin/bash

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets/eval_luoxin
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
ROOT_LOG=/data/datasets/jchen293/logs/exp/llava_eval

run_mmbench() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    SPLIT="mmbench_dev_20230712"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/mmbench/$SPLIT.tsv \
            --answers-file $ROOT_DATA/eval/mmbench/answers/$SPLIT/$NAME.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        mkdir -p $ROOT_DATA/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file $ROOT_DATA/eval/mmbench/$SPLIT.tsv \
            --result-dir $ROOT_DATA/eval/mmbench/answers/$SPLIT \
            --upload-dir $ROOT_DATA/eval/mmbench/answers_upload/$SPLIT \
            --experiment $NAME
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err"
}

NAME=light-compression-mmb
grouping=avgpool1d
layer=16
stride=8
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_mmbench $GPU_ID $NAME $layer $stride $grouping $CKPT
