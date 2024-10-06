#!/bin/bash
#
#SBATCH --job-name=v1_1_v5_seed
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/v1_1_v5_seed.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/v1_1_v5_seed.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main
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
        --conv-mode vicuna_v1 &
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

# layer1=0
# stride1=1
# grouping1=none
# CKPT1=/datasets/jchen293/data/llava_datasets/zhongrui/vlm_synthetic_data/LLaVA/checkpoints/llava-v1.5-7b-syn-v1.4-v2
# NAME1=v_1_4_v2_jc

layer1=0
stride1=1
grouping1=none
CKPT=/datasets/jchen293/data/llava_datasets/zhongrui/vlm_synthetic_data/LLaVA/checkpoints/llava-v1.5-7b-syn-v1.1-v5
NAME=v1_1_v5

run_seed $NAME $layer1 $stride1 $grouping1 $CKPT
# run_seed $NAME2 $layer2 $stride2 $grouping2 $CKPT2
# run_seed $NAME3 $layer3 $stride3 $grouping3 $CKPT3
# run_seed $NAME4 $layer4 $stride4 $grouping4 $CKPT4
# run_seed $NAME5 $layer5 $stride5 $grouping5 $CKPT5
# run_seed $NAME6 $layer6 $stride6 $grouping6 $CKPT6
# run_seed $NAME7 $layer7 $stride7 $grouping7 $CKPT7
# run_seed $NAME8 $layer8 $stride8 $grouping8 $CKPT8
# run_seed $NAME9 $layer9 $stride9 $grouping9 $CKPT9
# run_seed $NAME10 $layer10 $stride10 $grouping10 $CKPT10
# run_seed $NAME11 $layer11 $stride11 $grouping11 $CKPT11
# run_seed $NAME12 $layer12 $stride12 $grouping12 $CKPT12
# run_seed $NAME13 $layer13 $stride13 $grouping13 $CKPT13





