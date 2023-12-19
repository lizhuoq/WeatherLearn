#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
module load cudnn/8.6.0_cuda11.x

source activate weather

python pangu.py --lead_time 24
