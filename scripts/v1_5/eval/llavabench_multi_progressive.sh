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
        --conv-mode vicuna_v1

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
NAME1=1dpool8_2_2layer2_2_16pivot1300_2600_3900_v2
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2_2layer2_2_16pivot1300_2600_3900_v2
layer1=16
stride1=8
grouping1=avgpool1d

NAME2=1dpool8_8_2layer2_16_16pivot1300_2600_3900_v2
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_8_2layer2_16_16pivot1300_2600_3900_v2
layer2=16
stride2=8
grouping2=avgpool1d

NAME3=1dpool8_2_2layer2_2_16pivot1300_2600_3900_v3
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2_2layer2_2_16pivot1300_2600_3900_v3
layer3=16
stride3=64
grouping3=avgpool1d

NAME5=1dpool8_8_2layer2_16_16pivot1300_2600_3900prog_v3
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_8_2layer2_16_16pivot1300_2600_3900prog_v3
layer5=16
stride5=64
grouping5=avgpool1d

NAME6=1dpool8layer2_16pivot1730_3460-v2-v2
CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8layer2_16pivot1730_3460_v2
layer6=16
stride6=4
grouping6=avgpool1d

NAME7=1dpool8_2layer2_2pivot1730_3460
CKPT7=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2layer2_2pivot1730_3460
layer7=16
stride7=16
grouping7=avgpool1d

NAME8=1dpool8_2layer2_2pivot1730_3460-v2-v2
CKPT8=$ROOT_WEIGHT/llava-v1.5-7b-finetune-stride-8,2,1-layer-2,2,0-grouping-avgpool1d-progressive-v2
layer8=16
stride8=16
grouping8=avgpool1d

NAME4=1dpool8_2layer2_2pivot1730_3460_v3-v2
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-1dpool8_2layer2_2pivot1730_3460_v3
layer4=16
stride4=16
grouping4=avgpool1d



# run_llavabench $layer1 $stride1 $grouping1 $NAME1 $CKPT1 0
# run_llavabench $layer2 $stride2 $grouping2 $NAME2 $CKPT2 1
# run_llavabench $layer3 $stride3 $grouping3 $NAME3 $CKPT3 2
# run_llavabench $layer4 $stride4 $grouping4 $NAME4 $CKPT4 3
# run_llavabench $layer5 $stride5 $grouping5 $NAME5 $CKPT5 4
# run_llavabench $layer6 $stride6 $grouping6 $NAME6 $CKPT6 5
# run_llavabench $layer7 $stride7 $grouping7 $NAME7 $CKPT7 6
run_llavabench $layer8 $stride8 $grouping8 $NAME8 $CKPT8 7

wait

