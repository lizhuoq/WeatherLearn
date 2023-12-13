#!/bin/bash

module load anaconda/2020.11
module load cuda/12.0
module load cudnn/8.9.2_cuda12.x

source activate weather

python train.py
