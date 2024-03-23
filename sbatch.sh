#!/bin/bash -l
#SBATCH --nodelist=ccvl30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --gpus-per-node=8

module purge
module load conda
conda activate llava_git
 
bash scripts/v1_5/pretrain_slurm_pool4layer16.sh
