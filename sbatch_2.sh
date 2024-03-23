#!/bin/bash -l
#SBATCH --job-name=pool8layer16
#SBATCH --gpus=6000_ada:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --gpus-per-node=8

module purge
module load conda
conda activate llava_git
 
bash scripts/v1_5/pretrain_slurm_pool8layer16.sh

# sbatch --gpus=6000_ada:8 scripts/v1_5/pretrain_slurm_pool8layer16.sh