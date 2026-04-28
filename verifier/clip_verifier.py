"""
clip_verifier.py — Person 3 (Verifier / XAI)

V2: CLIP Cosine Similarity Verifier
- Crop image region using predicted bounding box from OWL-ViT
- Compute cosine similarity between cropped image and text prompt via CLIP
- If similarity < threshold → HALLUCINATED (label=1)
- If similarity >= threshold → REAL DETECTION (label=0)

Usage:
  python verifier/clip_verifier.py --data_dir /content/data
"""

import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from transformers import CLIPProcessor, CLIPModel


class CLIPVerifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32", threshold=0.5):
        """Initialize CLIP model for image-text similarity verification."""
        print(f"Loading CLIP model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.threshold = threshold
        print("CLIP model loaded successfully!")

    def compute_similarity(self, image, prompt):
        """
        Compute cosine similarity between an image and a text prompt using CLIP.
        Returns a float between -1 and 1 (typically 0.1 to 0.4 for real matches).
        """
        try:
            inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Normalize embeddings
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                # Cosine similarity
                similarity = (image_embeds @ text_embeds.T).item()

            return similarity

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def verify(self, similarity_score):
        """Classify based on CLIP similarity score."""
        if similarity_score >= self.threshold:
            return 0  # Real detection
        else:
            return 1  # Hallucination

    def verify_batch(self, scores):
        """Classify a batch of similarity scores."""
        return [self.verify(s) for s in scores]


def crop_image(image_path, box):
    """
    Crop an image using the bounding box coordinates [xmin, ymin, xmax, ymax].
    Returns a PIL Image of the cropped region.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Parse box if it's a string representation
        if isinstance(box, str):
            box = ast.literal_eval(box)

        if box is None or box == 'None':
            return img  # Return full image if no box

        xmin, ymin, xmax, ymax = box

        # Clamp to image boundaries
        w, h = img.size
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(w, int(xmax))
        ymax = min(h, int(ymax))

        # Ensure valid crop
        if xmax <= xmin or ymax <= ymin:
            return img

        cropped = img.crop((xmin, ymin, xmax, ymax))
        return cropped

    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        return Image.open(image_path).convert("RGB")


def grid_search_clip_threshold(true_labels, similarities, thresholds=None):
    """Find optimal CLIP similarity threshold via Grid Search on F1."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.50, 0.005)

    results = []
    for t in thresholds:
        preds = [0 if s >= t else 1 for s in similarities]
        f1 = f1_score(true_labels, preds, zero_division=0)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        results.append({
            'threshold': round(t, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    return best_threshold, results_df


def plot_clip_f1_vs_threshold(results_df, best_threshold, save_path=None):
    """Plot CLIP F1-Score vs Similarity Threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['threshold'], results_df['f1'], label='F1-Score', color='#9C27B0', linewidth=2)
    ax.plot(results_df['threshold'], results_df['precision'], label='Precision', color='#4CAF50', linewidth=1, linestyle='--')
    ax.plot(results_df['threshold'], results_df['recall'], label='Recall', color='#FF9800', linewidth=1, linestyle='--')

    best_f1 = results_df.loc[results_df['threshold'] == best_threshold, 'f1'].values[0]
    ax.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.7, label=f'Best Threshold = {best_threshold}')
    ax.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
    ax.annotate(f'Best F1={best_f1:.3f}\nThreshold={best_threshold}',
                xy=(best_threshold, best_f1),
                xytext=(best_threshold + 0.05, best_f1 - 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    ax.set_xlabel('CLIP Cosine Similarity Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('V2: F1-Score vs CLIP Similarity Threshold (Grid Search)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.close()


def run_clip_verification(base_dir):
    """Main function: load detection results, compute CLIP similarity, run grid search."""
    input_csv = base_dir / 'full_detection_results.csv'
    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found. Run detector.py first!")
        return

    print(f"Loading detection results from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows.")

    # --- Initialize CLIP ---
    clip = CLIPVerifier()

    # --- Compute similarities ---
    print("\nComputing CLIP similarities (this may take a while)...")
    similarities = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = base_dir / row['image_path']
        prompt = row['prompt']
        box = row['pred_box']

        # Crop image using predicted bounding box
        cropped = crop_image(str(img_path), box)

        # Compute CLIP similarity
        sim = clip.compute_similarity(cropped, prompt)
        similarities.append(sim)

    df['clip_similarity'] = similarities

    # --- Grid Search ---
    print("\nRunning Grid Search for optimal CLIP threshold...")
    best_threshold, grid_results = grid_search_clip_threshold(
        true_labels=df['true_label'].values,
        similarities=df['clip_similarity'].values
    )
    print(f"Best CLIP threshold: {best_threshold}")

    # --- Plot ---
    plot_path = base_dir / 'v2_clip_f1_vs_threshold.png'
    plot_clip_f1_vs_threshold(grid_results, best_threshold, save_path=str(plot_path))

    # --- Apply best threshold ---
    df['v2_prediction'] = clip.verify_batch(df['clip_similarity'].values)
    # Update threshold to best
    clip.threshold = best_threshold
    df['v2_prediction'] = clip.verify_batch(df['clip_similarity'].values)
    df['v2_threshold'] = best_threshold

    # --- Save results ---
    output_csv = base_dir / 'v2_clip_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # --- Summary ---
    f1 = f1_score(df['true_label'], df['v2_prediction'], zero_division=0)
    precision = precision_score(df['true_label'], df['v2_prediction'], zero_division=0)
    recall = recall_score(df['true_label'], df['v2_prediction'], zero_division=0)

    print(f"\n=== V2 CLIP Verifier Results ===")
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
        sub_f1 = f1_score(subset['true_label'], subset['v2_prediction'], zero_division=0)
        avg_sim = subset['clip_similarity'].mean()
        print(f"  {neg_type:10s}: F1={sub_f1:.4f}, Avg Similarity={avg_sim:.4f} ({len(subset)} samples)")

    # Save grid search
    grid_csv = base_dir / 'v2_grid_search.csv'
    grid_results.to_csv(grid_csv, index=False)
    print(f"\nGrid search results saved to: {grid_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2: CLIP Cosine Similarity Verifier")
    parser.add_argument('--data_dir', type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    base_dir = config.get_base_dir(args.data_dir)
    run_clip_verification(base_dir)
