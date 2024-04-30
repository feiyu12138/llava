#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name=llava-v1.5-7b-fastv-layer2-rank72-retrain
rank=72
k=2
ckpt=/data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-fastv-rank-72-k-2
python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank $rank \
    --fast-v-agg-layer $k 


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
