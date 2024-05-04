#!/bin/bash
#
#SBATCH --job-name=mme
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer2uvpe1d_mme.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool8layer2uvpe1d_mme.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=main

export CUDA_VISIBLE_DEVICES=0

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

layer=2
stride=8
grouping=avgpool1d
unified_vpe=True
name=llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d-unifiedvpe-True-retrain
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-8-layer-2-grouping-avgpool1d-unified_vpe-True
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --unified_vpe $unified_vpe

cd $ROOT_DATA/eval_luoxin/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name > ./eval_result/$name.txt
