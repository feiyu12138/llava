#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name=vcc_16_3_2
grouping=attn
stride=16
layer=2
num_fine_blocks=3
viz_assign=True
savedir=./viz_assign/$name-mme
ckpt=/home/lye21/llava_git/llava/playground/data/checkpoint/llava-v1.5-7b-stride-16-layer-2-grouping-attn-num_fine_block-3
python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping attn \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
