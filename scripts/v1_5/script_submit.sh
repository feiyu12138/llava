#!/bin/bash
#
#SBATCH --job-name=ptn
#SBATCH --error=/datasets/jchen293/logs/exp/llava/pt_none1.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/pt_none1.out
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --exclude=ccvl[14,33-38]
#SBATCH --mail-type=END
#SBATCH --mail-user=jchen293@jh.edu

cd /home/jchen293/code/llava_git/llava
for i in {1..50}
do
   bash scripts/v1_5/script_fresh.sh
   sleep 1h
done
