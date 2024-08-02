#!/bin/bash
#
#SBATCH --job-name=combine_llavaw
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/combine_llavaw.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/combine_llavaw.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=intern

module purge
module load conda
conda activate llava_git

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

export OPENAI_API_KEY=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

run_llavaw(){
    local GPU_ID=$1
    local NAME=$2
    local CKPT=$3
    local layer=$4
    local stride=$5
    local grouping=$6

    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "

    python -m llava.eval.model_vqa \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
        --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --layer $layer \
        --stride $stride \
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
    " > "/datasets/jchen293/logs/exp/llava_eval/${NAME}-llavaw.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${NAME}-llavaw.err" &
}

NAME1=1dpool2layer2
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-stride-2-layer-2-grouping-avgpool1d
layer1=2
stride1=2
grouping1=avgpool1d

NAME2=1dpool2layer2-v2
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-stride-2-layer-2-grouping-avgpool1d-v2
layer2=2
stride2=2
grouping2=avgpool1d

NAME3=1dpool16layer2
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-2-grouping-avgpool1d
layer3=16
stride3=2
grouping3=avgpool1d

NAME4=1dpool16layer2-v2
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-2-grouping-avgpool1d-v2
layer4=16
stride4=2
grouping4=avgpool1d

# NAME5=1dpool64layer16
# CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-16-grouping-avgpool1d
# layer5=16
# stride5=64
# grouping5=avgpool1d

# NAME6=1dpool4layer16
# CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-16-grouping-avgpool1d
# layer6=16
# stride6=4
# grouping6=avgpool1d

# NAME7=1dpool16layer16-wotrain
# CKPT7=$ROOT_WEIGHT/llava-v1.5-7b-reprod
# layer7=16
# stride7=16
# grouping7=avgpool1d

# NAME8=1dpool16layer16-rmask
# CKPT8=$ROOT_WEIGHT/llava-v1.5-7b-reprod
# layer8=16
# stride8=16
# grouping8=block_random_drop

run_llavaw 0 $NAME1 $CKPT1 $layer1 $stride1 $grouping1
run_llavaw 1 $NAME2 $CKPT2 $layer2 $stride2 $grouping2
run_llavaw 2 $NAME3 $CKPT3 $layer3 $stride3 $grouping3
run_llavaw 3 $NAME4 $CKPT4 $layer4 $stride4 $grouping4
# run_llavaw 4 $NAME5 $CKPT5 $layer5 $stride5 $grouping5
# run_llavaw 5 $NAME6 $CKPT6 $layer6 $stride6 $grouping6
# run_llavaw 6 $NAME7 $CKPT7 $layer7 $stride7 $grouping7
# run_llavaw 7 $NAME8 $CKPT8 $layer8 $stride8 $grouping8

wait
