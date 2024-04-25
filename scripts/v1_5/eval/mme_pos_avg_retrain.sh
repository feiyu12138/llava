#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
name=llava-v1.5-7b-stride-1-layer-0-grouping-pos_avg-retrain
CKPT="/data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-stride-1-layer-0-grouping-pos_avg"
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping pos_avg \
    --layer 0 
    # --viz


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
