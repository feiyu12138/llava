#!/bin/bash
#
#SBATCH --job-name=gqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/pool64layer0prog4600_gqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/pool64layer0prog4600_gqa.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main

module purge
module load conda
conda activate llava_git

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint


SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval_luoxin/eval/gqa/data"

unified_vpe=False
grouping=avgpool1d
stride=64
layer=0
pivot=4600
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe-progressive-$pivot
name=pool64layer0prog4600

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $ckpt \
        --question-file $ROOT_DATA/eval_luoxin/eval/gqa/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/gqa/images \
        --answers-file $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1  &
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

sleep 2d
