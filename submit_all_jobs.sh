#!/bin/bash
# Submit all 4 DPR experiments to DTU HPC
mkdir -p outputs
mkdir -p checkpoints

echo "Submitting Experiment 1 (Baseline Tuned: lr=3e-4, 25 epochs)..."
bsub < jobs/exp1_baseline.sh

echo "Submitting Experiment 2 (Compact: 3 layers, ffd=256, 30 epochs)..."
bsub < jobs/exp2_compact.sh

echo "Submitting Experiment 3 (Regularized: dropout=0.25, weight_decay=0.1)..."
bsub < jobs/exp3_regularized.sh

echo "Submitting Experiment 4 (LR 1e-4 Early Stop: 20 epochs)..."
bsub < jobs/exp4_lr1e4.sh

echo ""
echo "All 4 jobs submitted to DTU HPC GPU queue!"
echo "Track status with: bstat"
echo "When done, compare results with: python3 compare_results.py"
