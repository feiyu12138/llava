#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-8,8,1-layer-2,16,0-grouping-avgpool1d-progressive-v2
NAME=1dstride-8,8,1-layer-2,16,0-grouping-avgpool1d-progressive-v2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/vqav2/test2015 \
        --answers-file $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $NAME

