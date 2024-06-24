#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
CKPT=llava-v1.5-7b-finetune-qformer-linearproj
HASQF=True
export OPENAI_API_KEY=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

# python -m llava.eval.model_vqa \
#     --model-path $ROOT_WEIGHT/$CKPT \
#     --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
#     --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --has_qformer $HASQF \
#     --conv-mode vicuna_v1

# mkdir -p $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$CKPT.jsonl \
    --output \
        $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$CKPT.jsonl

python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$CKPT.jsonl
