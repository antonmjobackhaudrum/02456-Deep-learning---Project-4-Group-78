#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J dpr_exp3
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/exp3_regularized.out
#BSUB -e outputs/exp3_regularized.err

# Load modules
module load python3/3.10.13
module load cuda/12.1

unset PYTHONHOME
unset PYTHONPATH

mkdir -p outputs
mkdir -p checkpoints/exp3_regularized

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo "=========================================="
echo "Running Experiment 3: High Regularization"
echo "Params: lr=3e-4, weight_decay=0.1, dropout=0.25, epochs=30"
echo "=========================================="

# 1. Train
python3 train.py \
    device=cuda \
    training.lr=3e-4 \
    training.num_epochs=30 \
    training.weight_decay=0.1 \
    training.checkpoint_dir="checkpoints/exp3_regularized" \
    model.pooling_type=mean \
    model.n_layers=6 \
    model.feed_forward_dim=768 \
    model.dropout=0.25

# 2. Evaluate
python3 main.py \
    retriever=dpr \
    device=cuda \
    top_k=10 \
    model.pooling_type=mean \
    model.n_layers=6 \
    model.feed_forward_dim=768 \
    model.dropout=0.25 \
    model.query_encoder_weights_path="checkpoints/exp3_regularized/query_encoder.pt" \
    model.document_encoder_weights_path="checkpoints/exp3_regularized/document_encoder.pt" \
    retriever.index_path="checkpoints/exp3_regularized/document_embeddings.pt"
