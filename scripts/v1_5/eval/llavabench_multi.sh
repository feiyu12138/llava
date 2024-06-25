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
    export OPENAI_API_KEY=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
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

    python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl > $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/review_result/$NAME.txt
    "  &
}

NAME2=1dpool8layer16-v1-v2
grouping2=avgpool1d
layer2=16
stride2=8
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-16-grouping-avgpool1d

NAME3=1dpool8layer16-v2-v2
grouping3=avgpool1d
layer3=16
stride3=8
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-16-grouping-avgpool1d-v2

NAME4=1dpool8layer16-v3-v2
grouping4=avgpool1d
layer4=16
stride4=8
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-16-grouping-avgpool1d-v3

NAME5=1dpool64layer2-v2
grouping5=avgpool1d
layer5=2
stride5=64
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-16-grouping-avgpool1d-v2

# run_llavabench $layer1 $stride1 $grouping1 $NAME1 $CKPT1 0
run_llavabench $layer2 $stride2 $grouping2 $NAME2 $CKPT2 1
run_llavabench $layer3 $stride3 $grouping3 $NAME3 $CKPT3 2
run_llavabench $layer4 $stride4 $grouping4 $NAME4 $CKPT4 3
# run_llavabench $layer5 $stride5 $grouping5 $NAME5 $CKPT5 4

wait



