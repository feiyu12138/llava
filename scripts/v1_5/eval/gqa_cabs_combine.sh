#!/bin/bash

LLAVA_HOME=~/code/llava
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_gqa(){
    local CKPT=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local NAME=$5
    # for IDX in $(seq 0 $((CHUNKS-1))); do
    #     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
    #         --model-path $CKPT \
    #         --question-file $ROOT_DATA/eval_luoxin/eval/gqa/$SPLIT.jsonl \
    #         --image-folder $ROOT_DATA/eval_luoxin/eval/gqa/images \
    #         --answers-file $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
    #         --num-chunks $CHUNKS \
    #         --chunk-idx $IDX \
    #         --temperature 0 \
    #         --conv-mode vicuna_v1 \
    #         --grouping $grouping \
    #         --stride $stride \
    #         --layer $layer &
    # done

    # wait

    output_file=$ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME/merge.jsonl

    # # Clear out the output file if it exists.
    # > "$output_file"

    # # Loop through the indices and concatenate each file.
    # for IDX in $(seq 0 $((CHUNKS-1))); do
    #     cat $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    # done

    python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

    cd $GQADIR
    python eval/eval.py --tier testdev_balanced
    cd $LLAVA_HOME
}

layer1=2
stride1=4
name1=cabs_layer2_stride4
grouping1=cabstractor
ckpt1=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-4-layer-2-grouping-cabstractor

layer2=2
stride2=4
grouping2=Convabstractor
name2=conv_layer2_stride4
ckpt2=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-4-layer-2-grouping-Convabstractor

layer3=16
stride3=1
grouping3=cabstractor
name3=cabspool4_4_2layer2_16_16
ckpt3=$ROOT_WEIGHT/llava-v1.5-7b-cabspool4_4_2layer2_16_16pivot1300_2600_3900prog

run_gqa $ckpt1 $layer1 $stride1 $grouping1 $name1
run_gqa $ckpt2 $layer2 $stride2 $grouping2 $name2
run_gqa $ckpt3 $layer3 $stride3 $grouping3 $name3


