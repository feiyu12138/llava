#!/bin/bash
#
#SBATCH --job-name=1dpool64layer8_combined
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/1dpool64layer8_combined.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/1dpool64layer8_combined.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-stride-64-layer-8-grouping-avgpool1d
NAME=1dpool64layer8
layer=8
stride=64
grouping=avgpool1d

run_mmbench_cn() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
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
            --layer $layer \
            --stride $stride \
            --grouping $grouping 

        mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
            --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
            --experiment $NAME
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_mmbench() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    SPLIT="mmbench_dev_20230712"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --answers-file $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT/$NAME.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        mkdir -p $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/mmbench/$SPLIT.tsv \
            --result-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers/$SPLIT \
            --upload-dir $ROOT_DATA/eval_luoxin/eval/mmbench/answers_upload/$SPLIT \
            --experiment $NAME
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_mme() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
            --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        cd $ROOT_DATA/eval_luoxin/eval/MME
        python convert_answer_to_mme.py --experiment $NAME
        cd eval_tool
        python calculation.py --results_dir answers/$NAME > ./eval_result/$NAME.txt
        cd ~/llava_git/llava
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_mmvet() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/mm-vet/images \
            --answers-file $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        mkdir -p $ROOT_DATA/eval_luoxin/eval/mm-vet/results
        python scripts/convert_mmvet_for_eval.py \
            --src $ROOT_DATA/eval_luoxin/eval/mm-vet/answers/$NAME.jsonl \
            --dst $ROOT_DATA/eval_luoxin/eval/mm-vet/results/$NAME.json
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_pope() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/pope/val2014 \
            --answers-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        python llava/eval/eval_pope.py \
            --annotation-dir $ROOT_DATA/eval_luoxin/eval/pope/coco \
            --question-file $ROOT_DATA/eval_luoxin/eval/pope/llava_pope_test.jsonl \
            --result-file $ROOT_DATA/eval_luoxin/eval/pope/answers/$NAME.jsonl
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_sqa() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_science \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/scienceqa/llava_test_CQM-A.json \
            --image-folder $ROOT_DATA/eval_luoxin/eval/scienceqa/ScienceQA/test \
            --answers-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$NAME.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        python llava/eval/eval_science_qa.py \
            --base-dir $ROOT_DATA/eval_luoxin/eval/scienceqa \
            --result-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$NAME.jsonl \
            --output-file $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$NAME-output.jsonl \
            --output-result $ROOT_DATA/eval_luoxin/eval/scienceqa/answers/$NAME-result.json
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_textvqa() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/textvqa/train_images \
            --answers-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        python -m llava.eval.eval_textvqa \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file $ROOT_DATA/eval_luoxin/eval/textvqa/answers/$NAME.jsonl
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_vizwiz() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/vizwiz/test \
            --answers-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping

        python scripts/convert_vizwiz_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
            --result-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$NAME.jsonl \
            --result-upload-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers_upload/$NAME.json
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

run_llavabench(){
    local GPU_ID=$1
    local LOG_PREFIX=$2
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
    export OPENAI_API_KEY=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
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
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" 
}

run_seed(){
    local GPU_ID=$1
    local LOG_PREFIX=$2
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval_luoxin/eval/seed_bench/llava-seed-bench-img.jsonl \
            --image-folder $ROOT_DATA/eval_luoxin/eval/seed_bench \
            --answers-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping &
    done

    wait

    output_file=$ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval_luoxin/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # Evaluate
    python scripts/convert_seed_for_submission.py \
        --annotation-file $ROOT_DATA/eval_luoxin/eval/seed_bench/SEED-Bench.json \
        --result-file $output_file \
        --result-upload-file $ROOT_DATA/eval_luoxin/eval/seed_bench/answers_upload/$NAME.jsonl
}

run_mmbench_cn 0 "${NAME}-mmbench_cn" 
run_mmbench 1  "${NAME}-mmbench" 
run_mme 2 "${NAME}-mme"
run_mmvet 3 "${NAME}-mmvet"
run_pope 4 "${NAME}-pope"
run_sqa 5 "${NAME}-sqa"
run_textvqa 6 "${NAME}-textvqa"
run_vizwiz 7 "${NAME}-vizwiz"

wait

run_seed 0 "${NAME}-seed"
run_llavabench 0 "${NAME}-llavabench"