#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J dpr_training
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err

# Load modules (standard for DTU HPC)
module load python3/3.10.13
module load cuda/12.1

# Ensure output directory exists
mkdir -p outputs

# 1. Train the DPR encoders on GPU
python train.py device=cuda training.num_epochs=10

# 2. Build vector DB and evaluate dense retrieval
python main.py retriever=dpr device=cuda top_k=10
