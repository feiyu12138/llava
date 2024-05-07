#!/bin/bash
#
#SBATCH --job-name=textvqa
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/attnpool16layer2fine3cur_textvqa.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/attnpool16layer2fine3cur_textvqa.out
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
stride=16
grouping=attn
num_fine_blocks=3
unified_vpe=False

name=llava-v1.5-7b-stride-$stride-layer-$layer-fine-$num_fine_blocks-grouping-$grouping-num_fine_block-$num_fine_blocks
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-num_fine_block-$num_fine_blocks

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping \
    --num_fine_blocks $num_fine_blocks

python -m llava.eval.eval_textvqa \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl
