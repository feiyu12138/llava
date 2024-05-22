#!/bin/bash
#
#SBATCH --job-name=layer2_stride_8_random_drop_combine
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/layer2_stride_8_random_drop_combine.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/layer2_stride_8_random_drop_combined.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

CKPT=$ROOT_WEIGHT/llava-v1.5-7b-reprod
NAME=layer2_stride_8_random_drop
layer=2
stride=8
grouping=block_random_drop

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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

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
        # python -m llava.eval.model_vqa_loader \
        #     --model-path $CKPT \
        #     --question-file $ROOT_DATA/eval_luoxin/eval/MME/llava_mme.jsonl \
        #     --image-folder $ROOT_DATA/eval_luoxin/eval/MME/MME_Benchmark_release_version \
        #     --answers-file $ROOT_DATA/eval_luoxin/eval/MME/answers/$NAME.jsonl \
        #     --temperature 0 \
        #     --grouping $grouping \
        #     --stride $stride \
        #     --layer $layer \
        #     --conv-mode vicuna_v1

        cd $ROOT_DATA/eval_luoxin/eval/MME
        python convert_answer_to_mme.py --experiment $NAME
        cd eval_tool
        python calculation.py --results_dir answers/$NAME > ./eval_result/$NAME.txt
        cd ~/code/llava_git/llava
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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1 \

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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

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
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        python scripts/convert_vizwiz_for_submission.py \
            --annotation-file $ROOT_DATA/eval_luoxin/eval/vizwiz/llava_test.jsonl \
            --result-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers/$NAME.jsonl \
            --result-upload-file $ROOT_DATA/eval_luoxin/eval/vizwiz/answers_upload/$NAME.json
    " > "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.out" 2> "/datasets/jchen293/logs/exp/llava_eval/${LOG_PREFIX}.err" &
}

# run_mmbench_cn 0 "${NAME}-mmbench_cn" 
# run_mmbench 1  "${NAME}-mmbench" 
run_mme 0 "${NAME}-mme"
# run_mmvet 3 "${NAME}-mmvet"
# run_pope 4 "${NAME}-pope"
# run_sqa 5 "${NAME}-sqa"
# run_textvqa 6 "${NAME}-textvqa"
# run_vizwiz 7 "${NAME}-vizwiz"

wait
