#!/bin/bash
for stride in 2 4 8 16 32 64;
do for layer in 8 16 31;
do
python -m llava.eval.model_vqa \
    --model-path my-llava-1.5-7b \
    --question-file /home/lye21/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /data/jieneng/data/llava_datasets/eval/mmvet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-stride-$stride-layer-$layer.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --stride $stride \
    --layer $layer

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-stride-$stride-layer-$layer.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b-stride-$stride-layer-$layer.json
done
done

