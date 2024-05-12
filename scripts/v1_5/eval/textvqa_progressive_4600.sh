#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

grouping=none
stride=8
layer=2
unified_vpe=False
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-avgpool1d-unified_vpe-$unified_vpe-progressive-4600
name=1dpool8layer2progressive4600

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
    --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$name.jsonl
