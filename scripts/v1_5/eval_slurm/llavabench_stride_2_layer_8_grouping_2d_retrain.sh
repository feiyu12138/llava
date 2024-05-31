#!/bin/bash
#
#SBATCH --job-name=vccpool16layer2_llavaw
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/vccpool16layer2_llavaw.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/vccpool16layer2_llavaw.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=intern

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

module purge
module load conda
conda activate llava_git

export CUDA_VISIBLE_DEVICES=0
layer=16
stride=32
grouping=attn
num_fine_blocks=1

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-num_fine_block-$num_fine_blocks
NAME=vccpool16layer2

export OPENAI_API_KEY=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

# python -m llava.eval.model_vqa \
#     --model-path $CKPT \
#     --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
#     --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$name.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --stride $stride \
#     --layer $layer \
#     --grouping $grouping \
#     --num_fine_blocks $num_fine_blocks

# mkdir -p $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --output \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$name.jsonl

python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$name.jsonl > $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/review_result/$name-train.txt
