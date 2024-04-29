#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
name=llava-v1.5-7b-viz
ckpt=/data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping none \
    --stride 8 \
    --layer 2 \
    --viz \
    --viz_savepath stride-8-layer-2-grouping-avgpool1d \
    --num-fine-blocks 1 \
    --explore-prob 0.0


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b
