#!/bin/bash
#
#SBATCH --job-name=1dpool64_64_8layer2_16_16pivot1300_2600_3900progprog_gqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool64_64_8layer2_16_16pivot1300_2600_3900progprog_gqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool64_64_8layer2_16_16pivot1300_2600_3900progprog_gqa.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=60

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

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-1dpool64_64_8layer2_16_16pivot1300_2600_3900prog
NAME=1dpool64_64_8layer2_16_16pivot1300_2600_3900prog

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/gqa/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/gqa/images \
        --answers-file $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1  &
done

wait

output_file=$ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced

sleep 2d
