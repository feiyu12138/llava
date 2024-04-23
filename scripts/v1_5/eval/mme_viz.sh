#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
name=llava-v1.5-b7
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping pos_avg \
    --layer 0 
    # --viz


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b
