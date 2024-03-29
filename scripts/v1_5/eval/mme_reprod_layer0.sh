#!/bin/bash
CKPT="/home/jchen293/llava/checkpoints/llava-v1.5-7b-reprod"
export CUDA_VISIBLE_DEVICES=7
layer=-1
stride=4
grouping=avgpool2d
name=llava-v1.5-7b-reprod
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping   

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
