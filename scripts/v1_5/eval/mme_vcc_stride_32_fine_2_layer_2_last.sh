#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name=llava-v1.5-7b-vcc-layer-2-stride-32-fine-2-avg
grouping=attn
stride=32
layer=2
num_fine_blocks=2
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping attn \
    --stride $stride \
    --layer $layer \
    --num-fine-blocks $num_fine_blocks \
    --explore-prob 0.0 \


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
