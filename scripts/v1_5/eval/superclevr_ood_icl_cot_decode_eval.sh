#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
SPLIT="superclevr_questions_occlusion_ood"
name=llava-v1.5-7b
CLEVRDIR="./playground/data/eval/superclevr/data"
icl_file="./playground/data/eval/superclevr/ref_question.jsonl"
answer_file="./playground/data/eval/superclevr/answers/$SPLIT/$name/cot_decoding.jsonl"
output_file=./playground/data/eval/superclevr/answers/$SPLIT/$name/processed.jsonl
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_eval \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/superclevr/$SPLIT.jsonl \
        --image-folder /data/jieneng/data/llava_datasets/eval/superclevr/images \
        --answers-file $answer_file \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --cot-decoding \
        --num-branch 20 \
        --output-file $output_file
done

wait

output_file=./playground/data/eval/superclevr/answers/$SPLIT/$name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/superclevr/answers/$SPLIT/$name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $name --dir ./playground/data/eval/superclevr --test_split 

python scripts/convert_gqa_for_eval.py --src $output_file --dst $CLEVRDIR/ood_testdev_all_predictions.json

cd $CLEVRDIR
python eval/eval_easy.py --answer_file ../answer_superclevr_questions_occlusion_ood.jsonl --prediction_file ood_testdev_all_predictions.json > result/ood_icl_cot.txt

