#!/bin/bash
#
#SBATCH --job-name=multi_llavaw
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/multi_llavaw.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/multi_llavaw.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

export OPENAI_API_KEY=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

run_llavabench(){
    local CKPT=$1
    local NAME=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local GPU_ID=$6
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
    " &
}

NAME1=1dlayer2pool8-infer
layer1=2
stride1=8
grouping1=avgpool1d
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-reprod

NAME2=dhk-infer
layer2=2
stride2=8
grouping2=detach_hard_k_means
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-reprod

NAME3=rmasking-infer
layer3=2
stride3=8
grouping3=block_random_drop
CKPT3=$ROOT_WEIGHT/llava-v1.5-7b-reprod

NAME4=1dlayer2pool8
layer4=2
stride4=8
grouping4=avgpool1d
CKPT4=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-2-grouping-avgpool1d

NAME5=dhk-retrain
layer5=2
stride5=8
grouping5=detach_hard_k_means
CKPT5=$ROOT_WEIGHT/llava-v1.5-7b-stride-8-layer-2-grouping-detach_hard_k_means-unified_vpe-True

NAME6=rmasking-retrain
layer6=2
stride6=8
grouping6=block_random_drop
CKPT6=$ROOT_WEIGHT/llava-v1.5-7b-stride-16-layer-16-grouping-block_random_drop

run_llavabench $CKPT1 $NAME1 $layer1 $stride1 $grouping1 0
run_llavabench $CKPT2 $NAME2 $layer2 $stride2 $grouping2 1
run_llavabench $CKPT3 $NAME3 $layer3 $stride3 $grouping3 2
run_llavabench $CKPT4 $NAME4 $layer4 $stride4 $grouping4 3
run_llavabench $CKPT5 $NAME5 $layer5 $stride5 $grouping5 4
run_llavabench $CKPT6 $NAME6 $layer6 $stride6 $grouping6 5

wait