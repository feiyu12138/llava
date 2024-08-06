#!/bin/bash
#
#SBATCH --job-name=combined_seed
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/combined_seed.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/combined_seed.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=intern
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_seed(){
    local NAME=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local CKPT=$5

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
        --layer $layer \
        --stride $stride \
        --grouping $grouping &
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
        --result-upload-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers_upload/$NAME.jsonl \
        > $ROOT_DATA/eval_luoxin/eval/seed_bench/results/$NAME.txt

}

CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-reproduce
NAME1=multi-level-stride-4-infer
layer1=4,12,24
stride1=4
grouping1=avgpool1d

run_seed $NAME1 $layer1 $stride1 $grouping1 $CKPT1
# run_seed $NAME3 $layer3 $stride3 $grouping3 $CKPT3
# run_seed $NAME4 $layer4 $stride4 $grouping4 $CKPT4
# run_seed $NAME5 $layer5 $stride5 $grouping5 $CKPT5
# run_seed $NAME6 $layer6 $stride6 $grouping6 $CKPT6
# run_seed $NAME7 $layer7 $stride7 $grouping7 $CKPT7





