#!/bin/bash
#
#SBATCH --job-name=multi_gqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/multi_gqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/multi_gqa.out
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

run_gqa(){
    START=$(pwd)
    local CKPT=$1
    local NAME=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/gqa/$SPLIT.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/gqa/images \
            --answers-file $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping &
    done

    wait

    output_file=$ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

    cd $GQADIR
    python eval/eval.py --tier testdev_balanced > $ROOT_DATA/eval_luoxin/eval/gqa/answers/$SPLIT/$NAME.txt
    cd $START
}

NAME1=1dpool16layer2-v3
grouping1=avgpool1d
layer1=16
stride1=2
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-2-grouping-avgpool1d-v3

NAME2=1dpool2layer2-v3
grouping2=avgpool1d
layer2=2
stride2=2
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-stride-2-layer-2-grouping-avgpool1d-v3

# NAME3=1dlayer2pool8-v2
# grouping3=avgpool1d
# layer3=2
# stride3=8
# CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-v2

# NAME4=1dlayer2pool16-v2
# grouping4=avgpool1d
# layer4=2
# stride4=16
# CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-2-grouping-avgpool1d-v2

# NAME5=1dlayer2pool64-v2
# grouping5=avgpool1d
# layer5=2
# stride5=64
# CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-2-grouping-avgpool1d-v2

run_gqa $CKPT1 $NAME1 $layer1 $stride1 $grouping1
run_gqa $CKPT2 $NAME2 $layer2 $stride2 $grouping2
# run_gqa $CKPT3 $NAME3 $layer3 $stride3 $grouping3
# run_gqa $CKPT4 $NAME4 $layer4 $stride4 $grouping4
# run_gqa $CKPT5 $NAME5 $layer5 $stride5 $grouping5
