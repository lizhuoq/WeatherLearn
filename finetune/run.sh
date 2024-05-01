#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 96

python finetune_cpu.py
