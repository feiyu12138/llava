#!/bin/bash
#
#SBATCH --job-name=multi_reprod_seed
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/multi_reprod_seed.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/multi_reprod_seed.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=48

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
    local CKPT=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local NAME=$5

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/seed_bench/llava-seed-bench.jsonl \
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
        --result-upload-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers_upload/$NAME.jsonl

}

layer1=16
stride1=16
name1=1dpool16layer16_v2
grouping1=avgpool1d
ckpt1=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-avgpool1d-v2

layer2=16
stride2=64
grouping2=avgpool1d
name2=1dpool64layer16_v2
ckpt2=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-16-grouping-avgpool1d-v2

layer3=1
stride3=1
grouping3=avgpool1d
name3=none
ckpt3=$ROOT_WEIGHT/llava-v1.5-7b-stride-reprod-v2

run_seed $ckpt1 $layer1 $stride1 $grouping1 $name1
run_seed $ckpt2 $layer2 $stride2 $grouping2 $name2
run_seed $ckpt3 $layer3 $stride3 $grouping3 $name3



