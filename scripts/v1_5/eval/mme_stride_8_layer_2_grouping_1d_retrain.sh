#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
layer=2
stride=8
grouping=avgpool1d
name=llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-retrain
ckpt=/data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d
python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
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

python calculation.py --results_dir answers/$name > ./eval_result/$name.txt
