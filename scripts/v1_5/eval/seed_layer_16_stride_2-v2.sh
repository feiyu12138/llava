#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-2-layer-16-grouping-avgpool1d-v2
NAME=1dpool2layer16_v2

layer=16
stride=2
grouping=avgpool1d

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file $ROOT_DATA/eval_luoxin/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/seed_bench \
        --answers-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --layer $layer \
        --stride $stride \
        --grouping $grouping &
done

wait

output_file=$ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers_upload/llava-v1.5-13b.jsonl

