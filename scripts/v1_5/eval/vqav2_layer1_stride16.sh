#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
SPLIT="llava_vqav2_mscoco_test-dev2015"
name=llava-v1.5-7b-stride16-layer-1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $ckpt \
        --question-file $ROOT_DATA/eval_luoxin/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/vqav2/test2015 \
        --answers-file $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --layer 1 \
        --stride 16 \
        --grouping avgpool1d &
done

wait

output_file=$ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $name

