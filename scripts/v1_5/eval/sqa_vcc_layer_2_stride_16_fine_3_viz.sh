#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
name=llava-v1.5-7b-vcc-layer-2-stride-16-fine-3
grouping=attn
stride=16
layer=2
num_fine_blocks=3
viz_assign=True
savedir=./viz_assign/$name-sqa
ckpt=/home/lye21/llava_git/llava/playground/data/checkpoint/llava-v1.5-7b-stride-16-layer-2-grouping-attn-num_fine_block-3
python -m llava.eval.model_vqa_science \
    --model-path $ckpt \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/ScienceQA/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping attn \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$name-result.json
