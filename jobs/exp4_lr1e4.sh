#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J dpr_exp4
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/exp4_lr1e4.out
#BSUB -e outputs/exp4_lr1e4.err

# Load modules
module load python3/3.10.13
module load cuda/12.1

unset PYTHONHOME
unset PYTHONPATH

mkdir -p outputs
mkdir -p checkpoints/exp4_lr1e4

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo "=========================================="
echo "Running Experiment 4: LR 1e-4 with Early Stop"
echo "Params: lr=1e-4, epochs=20, layers=6, ffd=768, pooling=mean"
echo "=========================================="

# 1. Train
python3 train.py \
    device=cuda \
    training.lr=1e-4 \
    training.num_epochs=20 \
    training.checkpoint_dir="checkpoints/exp4_lr1e4" \
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
    model.query_encoder_weights_path="checkpoints/exp4_lr1e4/query_encoder.pt" \
    model.document_encoder_weights_path="checkpoints/exp4_lr1e4/document_encoder.pt" \
    retriever.index_path="checkpoints/exp4_lr1e4/document_embeddings.pt"
