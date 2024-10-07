#!/bin/bash
#
#SBATCH --job-name=multi_vqav2_r1
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/multi_vqav2_r1.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/multi_vqav2_r1.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=60

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

run_vqav2(){
    local CKPT=$1
    # local layer=$2
    # local stride=$3
    # local grouping=$4
    local NAME=$2
    for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/vqav2/test2015 \
        --answers-file $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    done

    wait

    output_file=$ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $NAME
}


name1=2dpool4_4_2layer2_16_16
ckpt1=$ROOT_WEIGHT/llava-v1.5-7b-1dpool4_4_2_layer2_16_16pivot1300_2600_3900


name2=1dpool16_16_4_layer2_16_16
ckpt2=$ROOT_WEIGHT/llava-v1.5-7b-1dpool16_16_4_layer2_16_16pivot1300_2600_3900


run_vqav2 $ckpt1 $name1
run_vqav2 $ckpt2 $name2



