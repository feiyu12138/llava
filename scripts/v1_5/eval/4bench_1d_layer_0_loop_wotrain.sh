#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
layer=0
grouping=avgpool1d
CKPT=liuhaotian/llava-v1.5-7b
export OPENAI_API_KEY=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
for stride in 4 16 64; do
name=stride-$stride-layer-$layer-grouping-$grouping
python -m llava.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /data/jieneng/data/llava_datasets/eval/scienceqa/ScienceQA/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping 

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$name-result.json

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer $layer \
    --stride $stride \
    --grouping $grouping 

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$name.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$name-retrain.json

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name > ./eval_result/$name.txt

cd ../../../../../

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer \
    --grouping $grouping

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context ./playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        ./playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl > playground/data/eval/llava-bench-in-the-wild/review_result/$name-train.txt
done
sleep 5d