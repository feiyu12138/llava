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

NAME2=2dpool4layer2
grouping2=avgpool2d
layer2=2
stride2=4
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-2dpool4layer2

NAME3=convpool4layer2
grouping3=Convabstractor
layer3=2
stride3=4
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-4-layer-2-grouping-Convabstractor

NAME4=cabspool4layer2
grouping4=cabstractor
layer4=2
stride4=4
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-4-layer-2-grouping-cabstractor

NAME5=cabspool4_4_2layer2_16_16
grouping5=cabstractor
layer5=16
stride5=1
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-cabspool4_4_2layer2_16_16pivot1300_2600_3900prog

NAME6=2dpool4_4_2layer2_16_16
CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-1dpool4_4_2_layer2_16_16pivot1300_2600_3900
layer6=16
stride6=1
grouping6=none

NAME7=1dpool16_16_4layer2_16_16
CKPT7=$ROOT_WEIGHT/llava-v1.5-7b-1dpool16_16_4_layer2_16_16pivot1300_2600_3900
layer7=16
stride7=1
grouping7=none

run_llavabench $layer2 $stride2 $grouping2 $NAME2 $CKPT2 1
run_llavabench $layer3 $stride3 $grouping3 $NAME3 $CKPT3 2
run_llavabench $layer4 $stride4 $grouping4 $NAME4 $CKPT4 3
run_llavabench $layer5 $stride5 $grouping5 $NAME5 $CKPT5 4
run_llavabench $layer6 $stride6 $grouping6 $NAME6 $CKPT6 5
run_llavabench $layer7 $stride7 $grouping7 $NAME7 $CKPT7 6

wait



