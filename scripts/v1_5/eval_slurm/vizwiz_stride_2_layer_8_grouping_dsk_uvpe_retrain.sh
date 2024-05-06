#!/bin/bash
#
#SBATCH --job-name=vizwiz
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/dskpool8layer2uvpe_vizwiz.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/dskpool8layer2uvpe_vizwiz.out
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
grouping=detach_soft_k_means
unified_vpe=True
name=llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe-retrain
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/vizwiz/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers//$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --unified_vpe $unified_vpe

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers//$name.jsonl \
    --result-upload-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers_upload//$name.json
