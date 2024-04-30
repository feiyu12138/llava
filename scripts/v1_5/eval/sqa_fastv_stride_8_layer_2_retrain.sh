#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
name=llava-v1.5-7b-fastv-layer2-rank72-retrain
rank=72
k=2
ckpt=/data/datasets/jchen293/weights/llava/checkpoint/llava-v1.5-7b-fastv-rank-72-k-2
python -m llava.eval.model_vqa_science \
    --model-path $ckpt \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /data/jieneng/data/llava_datasets/eval/scienceqa/ScienceQA/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank $rank \
    --fast-v-agg-layer $k 


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$name-result.json
