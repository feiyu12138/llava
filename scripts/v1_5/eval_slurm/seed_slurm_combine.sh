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

NAME2=2dpool4layer2
grouping2=avgpool2d
layer2=2
stride2=4
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-2dpool4layer2

NAME3=convpool4layer2
grouping3=Convabstractor
layer3=2
stride3=4
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-4-layer-2-grouping-Convabstractor

NAME4=cabspool4layer2
grouping4=cabstractor
layer4=2
stride4=4
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-4-layer-2-grouping-cabstractor

NAME5=cabspool4_4_2layer2_16_16
grouping5=cabstractor
layer5=16
stride5=1
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-cabspool4_4_2layer2_16_16pivot1300_2600_3900prog

NAME6=2dpool4_4_2layer2_16_16
CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-1dpool4_4_2_layer2_16_16pivot1300_2600_3900
layer6=16
stride6=1
grouping6=none

NAME7=1dpool16_16_4layer2_16_16
CKPT7=$ROOT_WEIGHT/llava-v1.5-7b-1dpool16_16_4_layer2_16_16pivot1300_2600_3900
layer7=16
stride7=1
grouping7=none

run_seed $NAME2 $layer2 $stride2 $grouping2 $CKPT2
run_seed $NAME3 $layer3 $stride3 $grouping3 $CKPT3
run_seed $NAME4 $layer4 $stride4 $grouping4 $CKPT4
run_seed $NAME5 $layer5 $stride5 $grouping5 $CKPT5
run_seed $NAME6 $layer6 $stride6 $grouping6 $CKPT6
run_seed $NAME7 $layer7 $stride7 $grouping7 $CKPT7





