#!/bin/bash
CUDA_VISIBLE_DEVICES=1,5,6,7
#,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="llava_ood_testdev_all"
GQADIR="./playground/data/eval/gqa/data"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_cot_decode.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --cot-decoding \
        --num-branch 20 &
        
done
wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge_cot_decode.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_cot_decode.jsonl >> "$output_file"
done

# python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/ood_testdev_all_predictions.json

# cd $GQADIR
# python eval/eval.py --tier ood_testdev_all > result/ood_testdev_all_llava-v1.5-7b.txt
# cd ../../../../..

