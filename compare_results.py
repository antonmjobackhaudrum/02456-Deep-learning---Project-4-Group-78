"""
Compare Recall@10 results across all experiments.
Run on HPC after jobs finish: python3 compare_results.py
"""
import glob
import re
import os

def parse_exp_log(log_path):
    with open(log_path, 'r', errors='ignore') as f:
        text = f.read()

    # Extract best val loss
    val_losses = [float(x) for x in re.findall(r'val loss:\s*([\d\.]+)', text)]
    best_val_loss = min(val_losses) if val_losses else None

    # Extract final train/val/test results
    train_recall = re.search(r"Train results:\s*\{'recall':\s*([\d\.]+)\}", text)
    val_recall = re.search(r"Val results:\s*\{'recall':\s*([\d\.]+)\}", text)
    test_recall = re.search(r"Test results:\s*\{'recall':\s*([\d\.]+)\}", text)

    return {
        'best_val_loss': f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A",
        'train_recall': f"{float(train_recall.group(1)):.4f}" if train_recall else "N/A",
        'val_recall': f"{float(val_recall.group(1)):.4f}" if val_recall else "N/A",
        'test_recall': f"{float(test_recall.group(1)):.4f}" if test_recall else "N/A",
    }

def main():
    experiments = [
        ("Exp 1: Baseline Tuned (lr=3e-4, 25ep)", "outputs/exp1_baseline.out"),
        ("Exp 2: Compact (3 layers, ffd=256)", "outputs/exp2_compact.out"),
        ("Exp 3: High Reg (drop=0.25, wd=0.1)", "outputs/exp3_regularized.out"),
        ("Exp 4: LR 1e-4 Early Stop (20ep)", "outputs/exp4_lr1e4.out"),
    ]

    header = f"| {'Experiment':<38} | {'Best Val Loss':<13} | {'Train Recall':<12} | {'Val Recall':<10} | {'Test Recall':<11} |"
    sep = f"|{'-'*40}|{'-'*15}|{'-'*14}|{'-'*12}|{'-'*13}|"

    print("\n" + "="*98)
    print("                      DPR EXPERIMENT COMPARISON SUMMARY")
    print("="*98)
    print(header)
    print(sep)

    for name, log_path in experiments:
        if os.path.exists(log_path):
            data = parse_exp_log(log_path)
            print(f"| {name:<38} | {data['best_val_loss']:<13} | {data['train_recall']:<12} | {data['val_recall']:<10} | {data['test_recall']:<11} |")
        else:
            print(f"| {name:<38} | {'(not finished)':<13} | {'-':<12} | {'-':<10} | {'-':<11} |")

    print(sep + "\n")

if __name__ == '__main__':
    main()
