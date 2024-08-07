#!/bin/bash

ROOT=/data/datasets/jchen293/weights/llava/checkpoint
PATH=llava-v1.5-7b-stride-reprod-v2

python -m llava.serve.cli --model-path $ROOT/$PATH --image-file "https://llava-vl.github.io/static/images/view.jpg"