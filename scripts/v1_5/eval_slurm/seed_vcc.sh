#!/bin/bash
#
#SBATCH --job-name=vccpool16layer2_seed
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/vccpool16layer2_seed.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/vccpool16layer2_seed.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=intern

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

module purge
module load conda
conda activate llava_git

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
layer=16
stride=32
grouping=attn
num_fine_blocks=1

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-num_fine_block-$num_fine_blocks
NAME=vccpool16layer2

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/seed_bench/llava-seed-bench-img.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/seed_bench \
        --answers-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_fine_blocks $num_fine_blocks &
done

wait

output_file=$ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers_upload/$NAME.jsonl

