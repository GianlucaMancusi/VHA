#!/bin/bash
#SBATCH --job-name=data_aug
#SBATCH --output=out.data_aug.txt
#SBATCH --error=err.data_aug.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1

source activate loco_env

python main.py --exp_name data_aug