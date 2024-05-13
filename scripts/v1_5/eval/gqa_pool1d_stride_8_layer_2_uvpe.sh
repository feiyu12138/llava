#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

name=llava-v1.5-7b-pool1d-stride-8-layer-2-grouping-avgpool1
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"
grouping=avgpool1d
stride=8
layer=2
unified_vpe=True

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file $ROOT_DATA/eval_luoxin/eval/gqa/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/gqa/images \
        --answers-file $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --grouping $grouping \
        --stride $stride \
        --layer $layer \
        --unified_vpe $unified_vpe &
done

wait

output_file=$ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
