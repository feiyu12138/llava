#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="llava_vqav2_mscoco_test-dev2015"

run_vqav2(){
    local CKPT=$1
    local NAME=$2
    local rank=$3
    local k=$4
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/vqav2/$SPLIT.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/vqav2/test2015 \
            --answers-file $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-fast-v True \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length 576 \
            --fast-v-attention-rank $rank \
            --fast-v-agg-layer $k &
    done

    wait

    output_file=$ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $NAME --dir $ROOT_DATA/eval_luoxin/eval/vqav2

}
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-fastv-rank-72-k-2
NAME1=fastv-rank-72-k-2
rank1=72
k1=2

CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-reprod
NAME2=fastv-rank-72-k-2-infer
rank2=72
k2=2

run_vqav2 $CKPT1 $NAME1 $rank1 $k1
run_vqav2 $CKPT2 $NAME2 $rank2 $k2




