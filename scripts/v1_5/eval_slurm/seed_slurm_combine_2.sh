#!/bin/bash
#
#SBATCH --job-name=combined_seed
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/combined_seed.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/combined_seed.out
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

layer1=0
stride1=1
grouping1=none
CKPT1=/datasets/jchen293/data/llava_datasets/zhongrui/vlm_synthetic_data/LLaVA/checkpoints/llava-v1.5-7b-syn-v1.4-v2
NAME1=v_1_4_v2

NAME2=v_1_4
layer1=0
stride1=1
grouping1=none
CKPT2=/datasets/jchen293/data/llava_datasets/zhongrui/vlm_synthetic_data/LLaVA/checkpoints/llava-v1.5-7b-syn-v1.4

NAME3=1dpool8layer2_16pivot1730_3460
layer3=2
stride3=4
grouping3=avgpool1d
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8layer2_16pivot1730_3460

NAME4=1dpool8_2_2layer2_2_16pivot1300_2600_3900prog
layer4=16
stride4=4
grouping4=avgpool1d
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2_2layer2_2_16pivot1300_2600_3900prog

NAME5=1dpool8_8_2layer2_16_16pivot1300_2600_3900prog
layer5=1
stride5=16
grouping5=avgpool1d
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_8_2layer2_16_16pivot1300_2600_3900prog

# NAME6=1dpool8_8_2layer2_16_16pivot1300_2600_3900prog
# layer6=16
# stride6=16
# grouping6=avgpool1d
# CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_8_2layer2_16_16pivot1300_2600_3900prog

# NAME7=rmasklayer16pool16_wotrain
# layer7=16
# stride7=16
# grouping7=block_random_drop
# CKPT7=$ROOT_WEIGHT/llava-v1.5-7b-reprod

# NAME8=dhklayer16pool16_wotrain
# layer8=16
# stride8=16
# grouping8=detach_hard_k_means
# CKPT8=$ROOT_WEIGHT/llava-v1.5-7b-reprod

# NAME9=dhklayer16pool16
# layer9=16
# stride9=16
# grouping9=detach_hard_k_means
# CKPT9=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-detach_hard_k_means

# NAME10=rmasklayer16pool16
# layer10=16
# stride10=16
# grouping10=block_random_drop
# CKPT10=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-block_random_drop

# NAME11=1dlayer16pool4-v2
# layer11=16
# stride11=4
# grouping11=avgpool1d
# CKPT11=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d-v2

# NAME12=1dlayer16pool4-v3
# layer12=16
# stride12=4
# grouping12=avgpool1d
# CKPT12=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d-v3

# NAME13=1dlayer16pool4-v4
# layer13=16
# stride13=4
# grouping13=avgpool1d
# CKPT13=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d-v4

run_seed $NAME1 $layer1 $stride1 $grouping1 $CKPT1
run_seed $NAME2 $layer2 $stride2 $grouping2 $CKPT2
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





