#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

run_seed(){
    local CKPT=$1
    local NAME=$2
    local k=$3
    local rank=$4
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/seed_bench/llava-seed-bench-img.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/seed_bench \
            --answers-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
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

CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-fastv-rank-72-k-2
NAME1=fastv-rank-72-k-2
rank1=72
k1=2

CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-reprod
NAME2=fastv-rank-72-k-2-infer
rank2=72
k2=2

run_seed $CKPT1 $NAME1 $k1 $rank1
run_seed $CKPT2 $NAME2 $k2 $rank2
