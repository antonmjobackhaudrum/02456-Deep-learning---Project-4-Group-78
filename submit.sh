#!/bin/sh
### General options
#BSUB -q gpua100
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

# Unset module variables that break virtual environments
unset PYTHONHOME
unset PYTHONPATH

# Ensure output directory exists
mkdir -p outputs

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# 1. Train the DPR encoders on GPU
python3 train.py device=cuda training.num_epochs=100

# 2. Build vector DB and evaluate dense retrieval
python3 main.py retriever=dpr device=cuda top_k=100
