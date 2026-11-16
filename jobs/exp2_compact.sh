#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J dpr_exp2
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/exp2_compact.out
#BSUB -e outputs/exp2_compact.err

# Load modules
module load python3/3.10.13
module load cuda/12.1

unset PYTHONHOME
unset PYTHONPATH

mkdir -p outputs
mkdir -p checkpoints/exp2_compact

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo "=========================================="
echo "Running Experiment 2: Compact Transformer"
echo "Params: lr=3e-4, epochs=30, layers=3, ffd=256, pooling=mean"
echo "=========================================="

# 1. Train
python3 train.py \
    device=cuda \
    training.lr=3e-4 \
    training.num_epochs=30 \
    training.checkpoint_dir="checkpoints/exp2_compact" \
    model.pooling_type=mean \
    model.n_layers=3 \
    model.feed_forward_dim=256 \
    model.dropout=0.1

# 2. Evaluate
python3 main.py \
    retriever=dpr \
    device=cuda \
    top_k=10 \
    model.pooling_type=mean \
    model.n_layers=3 \
    model.feed_forward_dim=256 \
    model.query_encoder_weights_path="checkpoints/exp2_compact/query_encoder.pt" \
    model.document_encoder_weights_path="checkpoints/exp2_compact/document_encoder.pt" \
    retriever.index_path="checkpoints/exp2_compact/document_embeddings.pt"
