#!/bin/bash
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

python eval_task_1.py