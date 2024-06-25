#!/bin/bash
#
#SBATCH --job-name=vqav2_multi
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/multi_vqav2.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/multi_vqav2.out
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

SPLIT="llava_vqav2_mscoco_test-dev2015"

run_vqav2(){
    local CKPT=$1
    local NAME=$2
    local stride=$3
    local layer=$4
    local grouping=$5
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

    python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $NAME --dir $ROOT_DATA/eval_luoxin/eval/vqav2
}

CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-reprod
NAME1=rmasking-infer
layer1=2
stride1=8
grouping1=block_random_drop

CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-reprod
NAME2=dhk-infer
layer2=2
stride2=8
grouping2=detach_hard_k_means

CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-reprod
NAME3=avgpool-infer
layer3=2
stride3=8
grouping3=avgpool1d

CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-block_random_drop
NAME4=rmasking-retrain
layer4=2
stride4=8
grouping4=block_random_drop

run_vqav2 $CKPT1 $NAME1 $stride1 $layer1 $grouping1
run_vqav2 $CKPT2 $NAME2 $stride2 $layer2 $grouping2
run_vqav2 $CKPT3 $NAME3 $stride3 $layer3 $grouping3
run_vqav2 $CKPT4 $NAME4 $stride4 $layer4 $grouping4



