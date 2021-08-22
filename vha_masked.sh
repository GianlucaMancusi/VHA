#!/bin/bash
#SBATCH --job-name=vha_masked
#SBATCH --output=slurm_out/out.vha_masked.txt
#SBATCH --error=slurm_out/err.vha_masked.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00


source activate loco_env

python main.py --exp_name vha_masked