#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J dpr_exp1
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/exp1_baseline.out
#BSUB -e outputs/exp1_baseline.err

# Load modules
module load python3/3.10.13
module load cuda/12.1

unset PYTHONHOME
unset PYTHONPATH

mkdir -p outputs
mkdir -p checkpoints/exp1_baseline

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo "=========================================="
echo "Running Experiment 1: Baseline Tuned"
echo "Params: lr=3e-4, epochs=25, layers=6, ffd=768, pooling=mean"
echo "=========================================="

# 1. Train
python3 train.py \
    device=cuda \
    training.lr=3e-4 \
    training.num_epochs=25 \
    training.checkpoint_dir="checkpoints/exp1_baseline" \
    model.pooling_type=mean \
    model.n_layers=6 \
    model.feed_forward_dim=768 \
    model.dropout=0.1

# 2. Evaluate
python3 main.py \
    retriever=dpr \
    device=cuda \
    top_k=10 \
    model.pooling_type=mean \
    model.n_layers=6 \
    model.feed_forward_dim=768 \
    model.query_encoder_weights_path="checkpoints/exp1_baseline/query_encoder.pt" \
    model.document_encoder_weights_path="checkpoints/exp1_baseline/document_encoder.pt" \
    retriever.index_path="checkpoints/exp1_baseline/document_embeddings.pt"
