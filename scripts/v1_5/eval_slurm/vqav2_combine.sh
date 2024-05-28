#!/bin/bash
#
#SBATCH --job-name=combine_vqav2
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/combine_vqav2.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/combine_vqav2.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=intern
#SBATCH --cpus-per-task=64

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

    local NAME=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local CKPT=$5

    for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/vqav2/test2015 \
        --answers-file $ROOT_DATA/eval_luoxin/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --layer $layer \
        --stride $stride \
        --grouping $grouping &
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


CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-detach_hard_k_means
NAME1=dhkpool16layer16
layer1=16
stride1=16
grouping1=detach_hard_k_means

CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-block_random_drop
NAME2=rmaskpool16layer16
layer2=16
stride2=16
grouping2=block_random_drop

CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d-v2
NAME3=1dlayer16pool4-v2
layer3=16
stride3=4
grouping3=avgpool1d

CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d-v3
NAME4=1dlayer16pool4-v3
layer4=16
stride4=4
grouping4=avgpool1d

CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d-v4
NAME5=1dlayer16pool4-v4
layer5=16
stride5=4
grouping5=avgpool1d

run_vqav2 $NAME1 $layer1 $stride1 $grouping1 $CKPT1
run_vqav2 $NAME2 $layer2 $stride2 $grouping2 $CKPT2
run_vqav2 $NAME3 $layer3 $stride3 $grouping3 $CKPT3
run_vqav2 $NAME4 $layer4 $stride4 $grouping4 $CKPT4
run_vqav2 $NAME5 $layer5 $stride5 $grouping5 $CKPT5



