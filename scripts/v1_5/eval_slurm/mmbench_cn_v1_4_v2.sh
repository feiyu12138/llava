#!/bin/bash
#
#SBATCH --job-name=v1_4_combined
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/v1_4_combined.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/v1_4_combined.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

CKPT=/datasets/jchen293/data/llava_datasets/zhongrui/vlm_synthetic_data/LLaVA/checkpoints/llava-v1.5-7b-syn-v1.4-v2
NAME=v1_4_v2

GPU_ID=0
LOG_PREFIX=$NAME-mmbench_cn
SPLIT="mmbench_dev_cn_20231003"
echo "Running mmbench_cn"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m llava.eval.model_vqa_mmbench \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
        --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$NAME.jsonl \
        --lang cn \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \

mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
    --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
    --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
    --experiment $NAME


# echo "Running experiments"
# run_mmbench_cn 1 "${NAME}-mmbench_cn" 
# # run_mmbench 1  "${NAME}-mmbench" 
# # run_mme 2 "${NAME}-mme"
# run_mmvet 0 "${NAME}-mmvet"
# run_pope 4 "${NAME}-pope"
# run_sqa 5 "${NAME}-sqa"
# run_textvqa 6 "${NAME}-textvqa"
# run_vizwiz 7 "${NAME}-vizwiz"

# wait
