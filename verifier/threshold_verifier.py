"""
threshold_verifier.py — Person 3 (Verifier / XAI)

V1: Confidence Threshold Verifier
- If max detection score < threshold → HALLUCINATED (label=1)
- If max detection score >= threshold → REAL DETECTION (label=0)
- Optimal threshold determined via Grid Search on F1-Score

Usage:
  python verifier/threshold_verifier.py --data_dir /content/data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ThresholdVerifier:
    def __init__(self, threshold=0.5):
        """Initialize with a confidence threshold."""
        self.threshold = threshold

    def verify(self, detection_score):
        """
        Classify a single prediction as genuine or hallucinated.
        Returns 0 (genuine) if score >= threshold, 1 (hallucinated) if below.
        """
        if detection_score >= self.threshold:
            return 0  # Real detection
        else:
            return 1  # Hallucination

    def verify_batch(self, scores):
        """Classify a batch of scores."""
        return [self.verify(s) for s in scores]


def grid_search_threshold(true_labels, pred_scores, thresholds=None):
    """
    Find the optimal confidence threshold by maximizing F1-Score.
    
    Parameters
    ----------
    true_labels : array-like
        Ground truth labels (0=genuine, 1=hallucination)
    pred_scores : array-like
        Confidence scores from OWL-ViT
    thresholds : array-like, optional
        List of thresholds to try. Default: 0.01 to 0.99 in steps of 0.01
    
    Returns
    -------
    best_threshold : float
    results_df : DataFrame with F1/Precision/Recall for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    results = []
    for t in thresholds:
        verifier = ThresholdVerifier(threshold=t)
        preds = verifier.verify_batch(pred_scores)

        f1 = f1_score(true_labels, preds, zero_division=0)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)

        results.append({
            'threshold': round(t, 3),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df


def plot_f1_vs_threshold(results_df, best_threshold, save_path=None):
    """Plot F1-Score vs Threshold with the optimal point highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['threshold'], results_df['f1'], label='F1-Score', color='#2196F3', linewidth=2)
    ax.plot(results_df['threshold'], results_df['precision'], label='Precision', color='#4CAF50', linewidth=1, linestyle='--')
    ax.plot(results_df['threshold'], results_df['recall'], label='Recall', color='#FF9800', linewidth=1, linestyle='--')

    # Highlight best threshold
    best_f1 = results_df.loc[results_df['threshold'] == best_threshold, 'f1'].values[0]
    ax.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.7, label=f'Best Threshold = {best_threshold}')
    ax.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
    ax.annotate(f'Best F1={best_f1:.3f}\nThreshold={best_threshold}',
                xy=(best_threshold, best_f1),
                xytext=(best_threshold + 0.1, best_f1 - 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('V1: F1-Score vs Confidence Threshold (Grid Search)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.close()


def run_threshold_verification(base_dir):
    """Main function: load detection results, run grid search, save verified results."""
    input_csv = base_dir / 'full_detection_results.csv'
    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found. Run detector.py first!")
        return

    print(f"Loading detection results from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows.")

    # --- Grid Search ---
    print("\nRunning Grid Search for optimal threshold...")
    best_threshold, grid_results = grid_search_threshold(
        true_labels=df['true_label'].values,
        pred_scores=df['pred_score'].values
    )
    print(f"Best threshold: {best_threshold}")

    # --- Plot ---
    plot_path = base_dir / 'v1_f1_vs_threshold.png'
    plot_f1_vs_threshold(grid_results, best_threshold, save_path=str(plot_path))

    # --- Apply best threshold ---
    verifier = ThresholdVerifier(threshold=best_threshold)
    df['v1_prediction'] = verifier.verify_batch(df['pred_score'].values)
    df['v1_threshold'] = best_threshold

    # --- Save results ---
    output_csv = base_dir / 'v1_threshold_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # --- Summary ---
    f1 = f1_score(df['true_label'], df['v1_prediction'], zero_division=0)
    precision = precision_score(df['true_label'], df['v1_prediction'], zero_division=0)
    recall = recall_score(df['true_label'], df['v1_prediction'], zero_division=0)

    print(f"\n=== V1 Threshold Verifier Results ===")
    print(f"  Optimal Threshold : {best_threshold}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1-Score          : {f1:.4f}")

    # --- Breakdown by negative type ---
    print(f"\n=== Breakdown by Negative Type ===")
    for neg_type in ['positive', 'hard', 'easy']:
        subset = df[df['negative_type'] == neg_type]
        if len(subset) == 0:
            continue
        sub_f1 = f1_score(subset['true_label'], subset['v1_prediction'], zero_division=0)
        print(f"  {neg_type:10s}: F1={sub_f1:.4f} ({len(subset)} samples)")

    # Save grid search results
    grid_csv = base_dir / 'v1_grid_search.csv'
    grid_results.to_csv(grid_csv, index=False)
    print(f"\nGrid search results saved to: {grid_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1: Confidence Threshold Verifier")
    parser.add_argument('--data_dir', type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    base_dir = config.get_base_dir(args.data_dir)
    run_threshold_verification(base_dir)
