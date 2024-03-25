#!/bin/bash
CKPT="/home/lye21/LLaVA/checkpoint/llava-v1.5-7b-reprod"
name=llava-v1.5-7b-reprod
python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$name.jsonl
