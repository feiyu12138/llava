#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROOT_DATA=/data/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/data/datasets/jchen293/weights/llava/checkpoint
export OPENAI_API_KEY=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
run_textvqa() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local NAME=$3
    local CKPT=$4
    local rank=$5
    local k=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
            --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-fast-v True \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length VISUAL_LENGTH \
            --fast-v-attention-rank $rank \
            --fast-v-agg-layer $k 

        python -m llava.eval.eval_textvqa \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$NAME.jsonl
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_pope() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local NAME=$3
    local CKPT=$4
    local rank=$5
    local k=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.eval_pope \
            --annotation-dir $ROOT_DATA/eval_luoxin/eval/pope/coco \
            --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
            --result-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$NAME.jsonl
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_vizwiz() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local NAME=$3
    local CKPT=$4
    local rank=$5
    local k=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/vizwiz/test \
            --answers-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-fast-v True \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length VISUAL_LENGTH \
            --fast-v-attention-rank $rank \
            --fast-v-agg-layer $k 

        python scripts/convert_vizwiz_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
            --result-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$NAME.jsonl \
            --result-upload-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers_upload/$NAME.json
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_mmbench_cn() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local NAME=$3
    local CKPT=$4
    local rank=$5
    local k=$6
    SPLIT="mmbench_dev_cn_20231003"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$NAME.jsonl \
            --lang cn \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-fast-v True \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length VISUAL_LENGTH \
            --fast-v-attention-rank $rank \
            --fast-v-agg-layer $k 

        mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
            --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
            --experiment $NAME
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_mmbench() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local NAME=$3
    local CKPT=$4
    local rank=$5
    local k=$6
    SPLIT="mmbench_dev_20230712"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$NAME.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-fast-v True \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length VISUAL_LENGTH \
            --fast-v-attention-rank $rank \
            --fast-v-agg-layer $k 

        mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
            --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
            --experiment $NAME
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_llavabench(){
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local NAME=$3
    local CKPT=$4
    local rank=$5
    local k=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
    export OPENAI_API_KEY=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

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
    " > "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/data/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

NAME1=fastv-rank-72-k-2
CKPT1=$ROOT_WEIGHT/llava-v1.5-7b-fastv-rank-72-k-2
# run_mmbench 0 $NAME1-mmbench_cn $NAME1 $CKPT1 72 2
# run_mmbench_cn 1 $NAME1-mmbench $NAME1 $CKPT1 72 2
# run_textvqa 2 $NAME1-textvqa $NAME1 $CKPT1 72 2
run_pope 3 $NAME1-pope $NAME1 $CKPT1 72 2
# run_vizwiz 4 $NAME1-vizwiz $NAME1 $CKPT1 72 2
# run_llavabench 5 $NAME1-llavabench $NAME1 $CKPT1 72 2

NAME2=fastv-rank-72-k-2-infer
CKPT2=$ROOT_WEIGHT/llava-v1.5-7b-reprod
# run_mmbench 0 $NAME2-mmbench_cn $NAME2 $CKPT2 72 2
# run_mmbench_cn 1 $NAME2-mmbench $NAME2 $CKPT2 72 2
# run_textvqa 2 $NAME2-textvqa $NAME2 $CKPT2 72 2
run_pope 3 $NAME2-pope $NAME2 $CKPT2 72 2
# run_vizwiz 4 $NAME2-vizwiz $NAME2 $CKPT2 72 2
# run_llavabench 5 $NAME2-llavabench $NAME2 $CKPT2 72 2

wait

