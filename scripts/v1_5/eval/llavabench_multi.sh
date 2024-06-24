#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
export OPENAI_API_KEY=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

run_llavabench(){
    local layer=$1
    local stride=$2
    local grouping=$3
    local NAME=$4
    local CKPT=$5
    local GPU_ID=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
    python -m llava.eval.model_vqa \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
        --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --stride $stride \
        --layer $layer \
        --grouping $grouping

    mkdir -p $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews

    python llava/eval/eval_gpt_review_bench.py \
        --question $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
        --context $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
        --output \
            $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl

    python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl > $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.txt
    "  &
}
NAME1=1dpool16layer1-v2
CKPT1="$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-1-grouping-avgpool1d"
layer1=1
stride1=16
grouping1=avgpool1d

NAME2=1dpool64layer1
CKPT2="$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-1-grouping-avgpool1d"
layer2=1
stride2=64
grouping2=avgpool1d

NAME3=reprod-v3
CKPT3="$ROOT_WEIGHT/llava-v1.5-7b-stride-reprod-v2"
layer3=16
stride3=16
grouping3=none

NAME4=1dpool64layer16-v3
CKPT4="$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-16-grouping-avgpool1d-v3"
layer4=16
stride4=64
grouping4=avgpool1d

run_llavabench $layer1 $stride1 $grouping1 $NAME1 $CKPT1 0
# run_llavabench $layer2 $stride2 $grouping2 $NAME2 $CKPT2 1
# run_llavabench $layer3 $stride3 $grouping3 $NAME3 $CKPT3 2
# run_llavabench $layer4 $stride4 $grouping4 $NAME4 $CKPT4 3

wait

