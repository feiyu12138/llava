#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

layer=0
stride=4
grouping=avgpool2d
name=1dlayer2pool64-v2
CKPT="/home/jchen293/llava/checkpoints/llava-v1.5-7b-finetune-stride-8,1-layer-2,0-grouping-avgpool1d-progressive-v2"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $CKPT \
#         --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/gqa/images \
#         --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 \
#         --stride $stride \
#         --layer $layer \
#         --grouping $grouping &
# done

# wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$name/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
