#!/bin/bash
#SBATCH --job-name=vha_d9_c3d
#SBATCH --output=slurm_out/out.vha_d9_c3d.txt
#SBATCH --error=slurm_out/err.vha_d9_c3d.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00


source activate loco_env

python main.py --exp_name vha_d_c3d_sphdiam9