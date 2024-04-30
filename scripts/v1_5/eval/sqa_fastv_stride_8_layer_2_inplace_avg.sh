#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
name=llava-v1.5-7b-fastv-layer2-rank72-inplace-avg
rank=72
k=2
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
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
