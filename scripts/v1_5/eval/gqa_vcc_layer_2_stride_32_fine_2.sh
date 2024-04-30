#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

name=llava-v1.5-7b-vcc-layer-2-stride-32-fine-2
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"
grouping=attn
stride=32
layer=2
num_fine_blocks=2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --grouping $grouping \
        --stride $stride \
        --num-fine-blocks $num_fine_blocks \
        --explore-prob 0.0 \
        --layer $layer &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
