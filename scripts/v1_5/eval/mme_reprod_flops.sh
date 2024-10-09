#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATA_PATH=/data/datasets/jchen293/data/llava_datasets/eval_luoxin/eval
CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
name=llava-v1.5-7b-reprod
layer=16
stride=2
python -m llava.eval.model_vqa_loader_flops \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file $DATA_PATH/MME/llava_mme.jsonl \
    --image-folder $DATA_PATH/MME/MME_Benchmark_release_version \
    --answers-file $DATA_PATH/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping avgpool1d

cd $DATA_PATH/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
