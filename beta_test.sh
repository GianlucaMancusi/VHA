#!/bin/bash
#SBATCH --job-name=beta_test
#SBATCH --output=log/beta_test/out.beta_test.txt
#SBATCH --error=log/beta_test/err.beta_test.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1

source activate loco_env

python main.py --exp_name beta_test