#!/bin/bash
#
#SBATCH --job-name=ptn
#SBATCH --error=/datasets/jchen293/logs/exp/llava/pt_none1.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/pt_none1.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --exclude=ccvl[14]

sleep 5d
