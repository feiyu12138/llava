#!/bin/bash
grouping=avgpool2d
for stride in 2 4 8; do
    for layer in 1 8 16 30; do
    name=llava-v1.5-b7
    python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$name-stride-$stride-layer-$layer.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --stride $stride \
        --layer $layer \
        --grouping $grouping

    cd ./playground/data/eval/MME

    python convert_answer_to_mme.py --experiment $name-stride-$stride-layer-$layer

    cd eval_tool

    python calculation.py --results_dir answers/$name-stride-$stride-layer-$layer > eval_result/$name-stride-$stride-layer-$layer.txt
    cd ../../../../..
    done
done
