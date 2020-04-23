#!/bin/bash
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

bert-score -r refs1.txt refs2.txt refs3.txt -c paths.hypo -s --lang en
