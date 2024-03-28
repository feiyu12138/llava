#!/bin/bash
#
#SBATCH --job-name=ptn
#SBATCH --error=/datasets/jchen293/logs/exp/llava/pt_none.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/pt_none.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --exclude=ccvl[14,33-38]

sleep 5d
