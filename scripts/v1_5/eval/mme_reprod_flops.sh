#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
name=llava-v1.5-7b-reprod
layer=16
stride=2
python -m llava.eval.model_vqa_loader_flops \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping avgpool1d

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
