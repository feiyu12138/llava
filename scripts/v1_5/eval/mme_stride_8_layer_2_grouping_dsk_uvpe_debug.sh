#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

layer=2
stride=8
grouping=detach_soft_k_means
unified_vpe=True
citer=1
name=llava-v1.5-7b-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe-retrain
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-$stride-layer-$layer-grouping-$grouping-unified_vpe-$unified_vpe
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
    --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
    --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping \
    --unified_vpe $unified_vpe \
    --citer $citer \
    --viz_assign True \
    --savedir ./viz_assign/$name-mme \

cd $ROOT_DATA/eval_luoxin/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name > ./eval_result/$name.txt
