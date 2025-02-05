#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

run_seed(){
    local CKPT=$1
    local NAME=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    # for IDX in $(seq 0 $((CHUNKS-1))); do
    #     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
    #         --model-path $CKPT \
    #         --question-file $ROOT_DATA/eval_luoxin/eval/seed_bench/llava-seed-bench-img.jsonl \
    #         --image-folder $ROOT_DATA/eval_luoxin/eval/seed_bench \
    #         --answers-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
    #         --num-chunks $CHUNKS \
    #         --chunk-idx $IDX \
    #         --temperature 0 \
    #         --conv-mode vicuna_v1 &
    # done

    wait

    output_file=$ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # Evaluate
    python scripts/convert_seed_for_submission.py \
        --annotation-file $ROOT_DATA/eval_luoxin/eval/seed_bench/SEED-Bench.json \
        --result-file $output_file \
        --result-upload-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers_upload/$NAME.jsonl
}

grouping2=avgpool1d
layer2=2
stride2=4
CKPT1=$ROOT_WEIGHT/llava-v1.5-13b-4stage
NAME1=13-4stage

grouping2=avgpool1d
layer2=2
stride2=8
CKPT2=$ROOT_WEIGHT/llava-v1.5-13b-reproduce
NAME2=13-reproduce

# NAME4=1dlayer2pool16-v2
# grouping4=avgpool1d
# layer4=2
# stride4=16
# CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-2-grouping-avgpool1d-v2

# NAME5=1dlayer2pool64-v2
# grouping5=avgpool1d
# layer5=2
# stride5=64
# CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-2-grouping-avgpool1d-v2

run_seed $CKPT1 $NAME1 $layer1 $stride1 $grouping1
# run_seed $CKPT2 $NAME2 $layer2 $stride2 $grouping2
# run_seed $CKPT3 $NAME3 $layer3 $stride3 $grouping3
# run_seed $CKPT4 $NAME4 $layer4 $stride4 $grouping4
# run_seed $CKPT5 $NAME5 $layer5 $stride5 $grouping5
