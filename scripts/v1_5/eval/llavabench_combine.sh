#!/bin/bash
#
#SBATCH --job-name=1dpool16layer0_llavaw
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool16layer0_llavaw.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool16layer0_llavaw.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=intern

# module purge
# module load conda
# conda activate llava_git
export CUDA_VISIBLE_DEVICES=0,1,2

ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-2-grouping-avgpool1d
NAME=1dpool16layer16-13b

layer=16
stride=16
grouping=avgpool1d

export OPENAI_API_KEY=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

run_llavaw(){
    local GPU_ID=$1
    local NAME=$2
    local CKPT=$3
    local layer=$4
    local stride=$5
    local grouping=$6

    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "

    # python -m llava.eval.model_vqa \
    #     --model-path $CKPT \
    #     --question-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/questions.jsonl \
    #     --image-folder $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/images \
    #     --answers-file $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
    #     --temperature 0 \
    #     --conv-mode vicuna_v1 \
    #     --layer $layer \
    #     --stride $stride \
    #     --grouping $grouping

    # mkdir -p $ROOT_DATA/eval_luoxin/eval/llava-bench-in-the-wild/reviews

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
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${NAME}-llavaw.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${NAME}-llavaw.err" &
}

NAME1=1dpool16layer2-v3
grouping1=avgpool1d
layer1=16
stride1=2
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-stride-2-layer-16-grouping-avgpool1d-v3

NAME2=1dpool2layer2-v3
grouping2=avgpool1d
layer2=2
stride2=2
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-stride-2-layer-2-grouping-avgpool1d-v3

CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-stride-4-layer-2-grouping-avgpool1d
NAME3=1dpool4layer2
layer3=2
stride3=4
grouping3=avgpool1d

run_llavaw 0 $NAME1 $CKPT1 $layer1 $stride1 $grouping1
run_llavaw 1 $NAME2 $CKPT2 $layer2 $stride2 $grouping2
# run_llavaw 2 $NAME3 $CKPT3 $layer3 $stride3 $grouping3

wait
