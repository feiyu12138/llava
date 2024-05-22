#!/bin/bash
#
#SBATCH --job-name=1dpool8_2_2layer2_2_16pivot1300_2600_3900prog_lw
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool8_2_2layer2_2_16pivot1300_2600_3900prog_lw.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool8_2_2layer2_2_16pivot1300_2600_3900prog_lw.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=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

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2_2layer2_2_16pivot1300_2600_3900prog
NAME=1dpool8_2_2layer2_2_16pivot1300_2600_3900prog


# python -m llava.eval.model_vqa \
#     --model-path $CKPT \
#     --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
#     --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
    --output \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl

python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl
