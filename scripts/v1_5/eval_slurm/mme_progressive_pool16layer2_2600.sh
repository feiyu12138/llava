#!/bin/bash
#
#SBATCH --job-name=1dpool16layer2progressive_mme
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool16layer2progressive_mme.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool16layer2progressive_mme.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=main

export CUDA_VISIBLE_DEVICES=0

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

unified_vpe=False
stride=16
layer=2
grouping=avgpool1d
ckpt=$ROOT_WEIGHT/1dpool16layer2progressive
name=1dpool16layer2progressive

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 

cd $ROOT_DATA/eval_luoxin/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name > ./eval_result/$name.txt
