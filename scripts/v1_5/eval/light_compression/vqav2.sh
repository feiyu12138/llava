#!/bin/bash
#
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets/eval_luoxin
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
ROOT_LOG=/data/datasets/jchen293/logs/exp/llava_eval

run_vqav2(){
    local CKPT=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local NAME=$5
    for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval/vqav2/test2015 \
        --answers-file $ROOT_DATA/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --layer $layer \
        --stride $stride \
        --grouping $grouping &
    done

    wait

    output_file=$ROOT_DATA/eval/vqav2/answers/$SPLIT/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $NAME --dir $ROOT_DATA/eval/vqav2
}

NAME=light-compression
grouping=avgpool1d
layer=16
stride=8
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME

run_vqav2 $CKPT $layer $stride $grouping $NAME