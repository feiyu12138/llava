#!/bin/bash
layer=16
stride=4
grouping=avgpool2d
name=stride-$stride-layer-$layer-grouping-$grouping
CKPT="/home/lye21/LLaVA/checkpoints/llava-v1.5-7b-$name"
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name > ./playground/data/eval/MME/eval_tool/eval_result/$name.txt
