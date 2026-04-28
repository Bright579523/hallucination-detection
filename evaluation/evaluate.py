"""
evaluate.py — Person 4 (Evaluation & Demo)

Computes evaluation metrics for V1 (Threshold) and V2 (CLIP) verifiers.
Generates: Confusion Matrices, ROC Curves, and comparative analysis.

Usage:
  python evaluation/evaluate.py --data_dir /content/data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_confusion_matrix(true_labels, predictions, title, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Genuine (0)', 'Hallucination (1)'],
                yticklabels=['Genuine (0)', 'Hallucination (1)'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(true_labels, v1_scores, v2_scores, save_path):
    """Plot ROC curves for V1 and V2 on the same figure."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # V1 ROC (invert scores: lower confidence = more likely hallucination)
    v1_inverted = 1 - np.array(v1_scores)
    fpr1, tpr1, _ = roc_curve(true_labels, v1_inverted)
    auc1 = roc_auc_score(true_labels, v1_inverted)
    ax.plot(fpr1, tpr1, label=f'V1 Threshold (AUC={auc1:.3f})', color='#2196F3', linewidth=2)

    # V2 ROC (invert scores: lower similarity = more likely hallucination)
    v2_inverted = 1 - np.array(v2_scores)
    fpr2, tpr2, _ = roc_curve(true_labels, v2_inverted)
    auc2 = roc_auc_score(true_labels, v2_inverted)
    ax.plot(fpr2, tpr2, label=f'V2 CLIP (AUC={auc2:.3f})', color='#9C27B0', linewidth=2)

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: V1 Threshold vs V2 CLIP', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ROC curve saved to: {save_path}")
    plt.close()

    return auc1, auc2


def compute_metrics(true_labels, predictions, name):
    """Compute and print classification metrics."""
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print(f"\n  --- {name} ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    return {'method': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def evaluate_by_negative_type(df, pred_col, method_name):
    """Evaluate metrics separately for each negative type."""
    print(f"\n  --- {method_name}: Breakdown by Negative Type ---")
    breakdown = []
    for neg_type in ['positive', 'hard', 'easy']:
        subset = df[df['negative_type'] == neg_type]
        if len(subset) == 0:
            continue
        f1 = f1_score(subset['true_label'], subset[pred_col], zero_division=0)
        prec = precision_score(subset['true_label'], subset[pred_col], zero_division=0)
        rec = recall_score(subset['true_label'], subset[pred_col], zero_division=0)
        acc = accuracy_score(subset['true_label'], subset[pred_col])
        print(f"  {neg_type:10s}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f} (n={len(subset)})")
        breakdown.append({
            'method': method_name, 'negative_type': neg_type,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'count': len(subset)
        })
    return breakdown


def run_evaluation(base_dir):
    """Main evaluation pipeline."""

    # --- Check for V1 results ---
    v1_csv = base_dir / 'v1_threshold_results.csv'
    v2_csv = base_dir / 'v2_clip_results.csv'

    has_v1 = v1_csv.exists()
    has_v2 = v2_csv.exists()

    if not has_v1 and not has_v2:
        print("ERROR: No verifier results found!")
        print("  Run threshold_verifier.py and/or clip_verifier.py first.")
        return

    print("=" * 60)
    print("  HALLUCINATION DETECTION — EVALUATION REPORT")
    print("=" * 60)

    all_metrics = []
    all_breakdowns = []

    # --- Evaluate V1 ---
    if has_v1:
        print(f"\nLoading V1 results from: {v1_csv}")
        df_v1 = pd.read_csv(v1_csv)
        metrics_v1 = compute_metrics(df_v1['true_label'], df_v1['v1_prediction'], 'V1: Confidence Threshold')
        all_metrics.append(metrics_v1)

        # Confusion Matrix
        plot_confusion_matrix(
            df_v1['true_label'], df_v1['v1_prediction'],
            f"V1 Confusion Matrix (Threshold={df_v1['v1_threshold'].iloc[0]})",
            str(base_dir / 'v1_confusion_matrix.png')
        )

        # Breakdown
        bd = evaluate_by_negative_type(df_v1, 'v1_prediction', 'V1: Threshold')
        all_breakdowns.extend(bd)

    # --- Evaluate V2 ---
    if has_v2:
        print(f"\nLoading V2 results from: {v2_csv}")
        df_v2 = pd.read_csv(v2_csv)
        metrics_v2 = compute_metrics(df_v2['true_label'], df_v2['v2_prediction'], 'V2: CLIP Similarity')
        all_metrics.append(metrics_v2)

        # Confusion Matrix
        plot_confusion_matrix(
            df_v2['true_label'], df_v2['v2_prediction'],
            f"V2 Confusion Matrix (Threshold={df_v2['v2_threshold'].iloc[0]})",
            str(base_dir / 'v2_confusion_matrix.png')
        )

        # Breakdown
        bd = evaluate_by_negative_type(df_v2, 'v2_prediction', 'V2: CLIP')
        all_breakdowns.extend(bd)

    # --- ROC Curve (only if both V1 and V2 available) ---
    if has_v1 and has_v2:
        print("\nGenerating ROC Curve comparison...")
        auc1, auc2 = plot_roc_curve(
            df_v1['true_label'].values,
            df_v1['pred_score'].values,
            df_v2['clip_similarity'].values,
            str(base_dir / 'roc_curve_comparison.png')
        )
        all_metrics[0]['roc_auc'] = auc1
        all_metrics[1]['roc_auc'] = auc2

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    summary_df = pd.DataFrame(all_metrics)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_csv = base_dir / 'evaluation_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")

    # Save breakdown
    if all_breakdowns:
        breakdown_df = pd.DataFrame(all_breakdowns)
        breakdown_csv = base_dir / 'evaluation_breakdown.csv'
        breakdown_df.to_csv(breakdown_csv, index=False)
        print(f"Breakdown saved to: {breakdown_csv}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V1 and V2 verifiers")
    parser.add_argument('--data_dir', type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    base_dir = config.get_base_dir(args.data_dir)
    run_evaluation(base_dir)
