#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="my-llava-1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"
stride=2
layer=16
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /data/jieneng/data/llava_datasets/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/stride-$stride-layer-$layer/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --stride $stride \
        --layer $layer &
done


# New loop variables
# for stride in 2 4 8 16 32 64; do
#     for layer in 8 16 31; do
#         for IDX in $(seq 0 $((CHUNKS-1))); do
#             CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m llava.eval.model_vqa_loader \
#                 --model-path liuhaotian/llava-v1.5-7b \
#                 --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#                 --image-folder /data/jieneng/data/llava_datasets/eval/vqav2/test2015 \
#                 --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/stride-$stride-layer-$layer/${CHUNKS}_${IDX}.jsonl \
#                 --num-chunks $CHUNKS \
#                 --chunk-idx $IDX \
#                 --temperature 0 \
#                 --conv-mode vicuna_v1 \
#                 --stride $stride \
#                 --layer $layer 
#         done

#         wait

#         output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/stride-$stride-layer-$layer/merge.jsonl

#         # Ensure output directory exists
#         mkdir -p "$(dirname "$output_file")"

#         # Clear out the output file if it exists.
#         > "$output_file"

#         # Loop through the indices and concatenate each file.
#         for IDX in $(seq 0 $((CHUNKS-1))); do
#             cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/stride-$stride-layer-$layer/${CHUNKS}_${IDX}.jsonl >> "$output_file"
#         done

#         python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --stride $stride --layer $layer
#     done
# done
