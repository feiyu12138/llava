#!/bin/bash
#
#SBATCH --job-name=multi_reprod_combine
#SBATCH --error=/datasets/jchen293/logs/exp/llava_eval/multi_reprod_combine.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava_eval/multi_reprod_combine.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=main

bash scripts/v1_5/eval_slurm/combined_layer16_64_v2.sh >/datasets/jchen293/logs/exp/llava_eval/1dpool64layer16v2.out 2>/datasets/jchen293/logs/exp/llava_eval/1dpool64layer16v2.err
bash scripts/v1_5/eval_slurm/combined_layer16_16_v2.sh >/datasets/jchen293/logs/exp/llava_eval/1dpool16layer16v2.out 2>/datasets/jchen293/logs/exp/llava_eval/1dpool16layer16v2.err
bash scripts/v1_5/eval_slurm/combined_reprod_v2.sh >/datasets/jchen293/logs/exp/llava_eval/reprodv2.out 2>/datasets/jchen293/logs/exp/llava_eval/reprodv2.err