#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

grouping=attn
stride=16
layer=8
num_fine_blocks=3
viz_assign=True
selector_type=user_token
name=vcc-layer-$layer-stride-$stride-fine-$num_fine_blocks-wotrain-jn
savedir=./viz_assign/$name-$selector_type-sqa
ckpt=/data/jieneng/huggingface/llava-v1.5-7b

python -m llava.eval.model_vqa_science \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $ROOT_DATA/eval_luoxin/eval/scienceqa/ScienceQA/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --grouping attn \
    --stride $stride \
    --layer $layer \
    --num_fine_blocks $num_fine_blocks \
    --explore_prob 0.0 \
    --viz_assign $viz_assign \
    --savedir $savedir \
    --selector_type $selector_type


python llava/eval/eval_science_qa.py \
    --base-dir $ROOT_DATA/eval_luoxin/eval/scienceqa \
    --result-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name.jsonl \
    --output-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-output.jsonl \
    --output-result $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$name-result.json
