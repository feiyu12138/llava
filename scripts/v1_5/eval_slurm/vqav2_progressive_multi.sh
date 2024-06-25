#!/bin/bash
#
#SBATCH --job-name=vqav2
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/vqav2_multi.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/vqav2_multi.out
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
    local NAME=$1
    local CKPT=$2
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
NAME1=pool8layer2_progressive_v2
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-8,1-layer-2,0-grouping-avgpool1d-progressive-v2

NAME2=pool8layer2_progressive_v3
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8layer2pivot2600_v3

NAME3=pool8layer2_16_progressive_v3
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8layer2_16pivot1730_3460_v3

NAME4=pool8_2layer2_progressive_v3
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2layer2_2pivot1730_3460_v3

NAME5=pool8_2_2layer2_2_16_progressive_v3
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2_2layer2_2_16pivot1300_2600_3900_v3

NAME6=pool8_8_2layer2_16_16_2_progressive_v3
CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2_2layer2_2_16pivot1300_2600_3900_v3

run_vqav2 $NAME1 $CKPT1
run_vqav2 $NAME2 $CKPT2
run_vqav2 $NAME3 $CKPT3
run_vqav2 $NAME4 $CKPT4
run_vqav2 $NAME5 $CKPT5
run_vqav2 $NAME6 $CKPT6




