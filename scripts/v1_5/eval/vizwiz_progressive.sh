#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

grouping=none
stride=8
layer=2
unified_vpe=False
ckpt=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-avgpool1d-unified_vpe-$unified_vpe-progressive
name=llava-v1.5-7b-progressive

python -m llava.eval.model_vqa_loader \
    --model-path $ckpt \
    --question-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/vizwiz/test \
    --answers-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
    --result-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$name.jsonl \
    --result-upload-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers_upload/$name.json
