#!/bin/bash
#
#SBATCH --job-name=layer2_16_16pool64_64_16_pivot_1730_3460_llavaw
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/layer2_16_16pool64_64_16_pivot_1730_3460_llavaw.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/layer2_16_16pool64_64_16_pivot_1730_3460_llavaw.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=intern
#SBATCH --exclude=ccvl[14]

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-1dpool_64_64_16layer_2_16_16pivot_1730_3460
NAME=1dpool64_64_16layer2_16_16pivot1730_3460prog

export OPENAI_API_KEY=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

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

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
