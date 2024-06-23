#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
ROOT_LOG=/data/datasets/jchen293/logs/exp/llava_eval

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-block_random_drop
NAME=1dpool16layer16rmask

layer=16
stride=16
grouping=block_random_drop

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --grouping $grouping \
        --stride $stride \
        --layer $layer &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
