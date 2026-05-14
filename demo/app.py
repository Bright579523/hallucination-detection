"""
app.py — Person 4 (Evaluation & Demo)

Interactive Gradio Web Demo for Hallucination Detection.
3-Tab Professional Dashboard: Live Demo | Evaluation Metrics | Architecture

Usage:
  python demo/app.py                        # Local demo
  python demo/app.py --share                # Public share link (Colab)
  python demo/app.py --data_dir /content/data  # Override data dir
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    CLIPProcessor, CLIPModel
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }

/* Hero */
.hero-section {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-section h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2rem !important; font-weight: 800 !important; margin-bottom: 4px !important;
}
.hero-section p { color: #b0b3c5 !important; font-size: 0.95rem; }

/* Verdict cards */
.verdict-hall {
    background: linear-gradient(135deg, #1a0000, #330000);
    border: 2px solid #ff4444; border-radius: 16px; padding: 20px;
    animation: pulse-r 2s ease-in-out infinite;
}
.verdict-gen {
    background: linear-gradient(135deg, #001a00, #003300);
    border: 2px solid #44ff44; border-radius: 16px; padding: 20px;
    animation: pulse-g 2s ease-in-out infinite;
}
@keyframes pulse-r {
    0%,100% { box-shadow: 0 0 20px rgba(255,68,68,0.3); }
    50% { box-shadow: 0 0 40px rgba(255,68,68,0.6); }
}
@keyframes pulse-g {
    0%,100% { box-shadow: 0 0 20px rgba(68,255,68,0.3); }
    50% { box-shadow: 0 0 40px rgba(68,255,68,0.6); }
}

/* Score containers */
.score-box {
    background: rgba(255,255,255,0.05); border-radius: 12px;
    padding: 14px 18px; margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.08);
}
.bar-bg {
    background: rgba(255,255,255,0.1); border-radius: 8px;
    height: 22px; overflow: hidden; margin-top: 6px;
}
.bar-red { height:100%; border-radius:8px; background:linear-gradient(90deg,#ff4444,#ff6b6b); }
.bar-green { height:100%; border-radius:8px; background:linear-gradient(90deg,#44ff44,#6bff6b); }
.bar-amber { height:100%; border-radius:8px; background:linear-gradient(90deg,#ffaa00,#ffd000); }

/* Info card */
.info-card {
    background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.25);
    border-radius: 12px; padding: 18px 22px; margin-top: 12px;
}
"""


# ---------------------------------------------------------------------------
# Detector class (backend logic — unchanged)
# ---------------------------------------------------------------------------
class HallucinationDetector:
    """Combined detector + verifier for the demo."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading OWL-ViT...")
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        ).to(self.device)
        self.owl_model.eval()

        print("Loading CLIP...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_model.eval()

        print("Models loaded!")

    def detect_and_verify(self, image, prompt, conf_threshold=0.15, clip_threshold=0.22):
        if image is None or not prompt.strip():
            return None, None, "<p style='color:#ff6b6b;'>⚠️ Please upload an image and enter a text prompt.</p>"

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # OWL-ViT detection
        texts = [[prompt]]
        inputs = self.owl_processor(text=texts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.owl_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owl_processor.image_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

        if len(results["scores"]) == 0:
            return image, None, "<p style='color:#ffd000;'>🤷 No detection — the model found nothing.</p>"

        best_idx = torch.argmax(results["scores"]).item()
        best_score = results["scores"][best_idx].item()
        best_box = results["boxes"][best_idx].tolist()

        v1_pass = best_score >= conf_threshold

        # CLIP verification
        xmin, ymin, xmax, ymax = [int(c) for c in best_box]
        w, h = image.size
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        cropped = image.crop((xmin, ymin, xmax, ymax)) if xmax > xmin and ymax > ymin else image

        clip_inputs = self.clip_processor(
            text=[prompt], images=cropped, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            clip_out = self.clip_model(**clip_inputs)
            img_emb = clip_out.image_embeds / clip_out.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = clip_out.text_embeds / clip_out.text_embeds.norm(dim=-1, keepdim=True)
            clip_sim = (img_emb @ txt_emb.T).item()

        v2_pass = clip_sim >= clip_threshold

        # Draw bounding box
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        is_hall = (not v1_pass) or (not v2_pass)
        color = "#FF4444" if is_hall else "#44FF44"
        for i in range(4):
            draw.rectangle([xmin - i, ymin - i, xmax + i, ymax + i], outline=color)
        label = f"{prompt} ({best_score:.2f})"
        tb = draw.textbbox((xmin, ymin - 20), label)
        draw.rectangle([tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=color)
        draw.text((xmin, ymin - 20), label, fill="white")

        # Build HTML
        html = self._html(prompt, best_score, conf_threshold, v1_pass,
                          clip_sim, clip_threshold, v2_pass, is_hall,
                          xmin, ymin, xmax, ymax)
        return annotated, cropped, html

    # ── HTML builder ──────────────────────────────────────
    def _html(self, prompt, owl, ct, v1p, clip, clt, v2p, is_h, x1, y1, x2, y2):
        vc = "verdict-hall" if is_h else "verdict-gen"
        em = "🚨" if is_h else "✅"
        vt = "HALLUCINATION DETECTED" if is_h else "GENUINE DETECTION"
        vc2 = "#ff6b6b" if is_h else "#6bff6b"
        desc = ("The model was likely misled by the text prompt."
                if is_h else "Both verifiers agree — the object matches the prompt.")

        def bar(score, thresh, passed):
            pct = min(score * 100, 100)
            tpct = min(thresh * 100, 100)
            cls = "bar-green" if passed else ("bar-amber" if score > thresh * 0.7 else "bar-red")
            st = "✅ PASS" if passed else "❌ FAIL"
            sc = "#44ff44" if passed else "#ff4444"
            return f"""
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:1.4rem;font-weight:700;">{score:.4f}</span>
                <span style="color:{sc};font-weight:700;">{st}</span>
            </div>
            <div class="bar-bg" style="position:relative;">
                <div class="{cls}" style="width:{pct}%;"></div>
                <div style="position:absolute;left:{tpct}%;top:0;height:100%;
                     border-left:2px dashed rgba(255,255,255,0.7);"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
                <span style="color:#888;font-size:0.75rem;">0.0</span>
                <span style="color:#aaa;font-size:0.75rem;">▲ Threshold: {thresh:.2f}</span>
                <span style="color:#888;font-size:0.75rem;">1.0</span>
            </div>"""

        return f"""
        <div class="{vc}">
            <div style="text-align:center;">
                <span style="font-size:2.8rem;">{em}</span>
                <h2 style="margin:6px 0 4px;font-size:1.5rem;color:{vc2};">{vt}</h2>
                <p style="color:#aaa;margin:0;font-size:0.85rem;">{desc}</p>
            </div>
        </div>
        <p style="color:#ccc;font-size:0.9rem;margin-top:14px;">
            🔍 Prompt: <strong style="color:#fff;">"{prompt}"</strong>
        </p>
        <div class="score-box">
            <h3 style="margin:0 0 6px;font-size:0.95rem;">🎯 V1: OWL-ViT Confidence</h3>
            <p style="color:#999;font-size:0.78rem;margin:0 0 6px;">
                How confident is OWL-ViT that this object exists?</p>
            {bar(owl, ct, v1p)}
        </div>
        <div class="score-box">
            <h3 style="margin:0 0 6px;font-size:0.95rem;">🧠 V2: CLIP Similarity</h3>
            <p style="color:#999;font-size:0.78rem;margin:0 0 6px;">
                Does the cropped region match the prompt? (See "What CLIP Sees")</p>
            {bar(clip, clt, v2p)}
        </div>
        <div style="background:rgba(255,255,255,0.03);border-radius:8px;
             padding:10px 14px;margin-top:8px;border:1px solid rgba(255,255,255,0.06);">
            <p style="color:#888;font-size:0.78rem;margin:0;">
                📐 Bounding Box: [{x1}, {y1}, {x2}, {y2}]</p>
        </div>"""


# ---------------------------------------------------------------------------
# Build the 3-Tab Dashboard
# ---------------------------------------------------------------------------
def build_demo(share=False, data_dir=None):
    print("Initializing models (this may take a minute)...")
    detector = HallucinationDetector()

    base_dir = config.get_base_dir(data_dir)

    def predict(image, prompt, ct, clt):
        return detector.detect_and_verify(image, prompt, ct, clt)

    # Helper to load evaluation images if they exist
    def load_eval_image(filename):
        path = base_dir / filename
        if path.exists():
            return str(path)
        return None

    theme = gr.themes.Soft(primary_hue="purple", neutral_hue="slate")

    with gr.Blocks(theme=theme, css=CUSTOM_CSS,
                   title="Hallucination Detection Dashboard") as demo:

        # ── Hero Header ──
        gr.HTML("""
        <div class="hero-section">
            <h1>🔬 Prompt Hallucination Detection</h1>
            <p style="font-size:1.05rem;color:#d0d3e5!important;margin-top:6px;">
                Detecting false object detections in OWL-ViT caused by misleading text prompts
            </p>
        </div>
        """)

        # ══════════════════════════════════════════════════
        #  3 TABS
        # ══════════════════════════════════════════════════
        with gr.Tabs():

            # ─── TAB 1: Live Demo ─────────────────────────
            with gr.TabItem("🚀 Live Demo Verifier"):
                with gr.Row(equal_height=False):
                    # Left: Inputs
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 📥 Input Settings")
                        input_image = gr.Image(label="📷 Upload Image", type="pil", height=280)
                        input_prompt = gr.Textbox(
                            label="✏️ Text Prompt",
                            placeholder='e.g. "a photo of a cat"',
                            info="Describe the object you want the model to find."
                        )
                        with gr.Accordion("⚙️ Threshold Settings", open=False):
                            gr.Markdown("*Higher = stricter verification (fewer false positives)*")
                            conf_slider = gr.Slider(0.01, 0.99, value=0.15, step=0.01,
                                                    label="V1: Confidence Threshold",
                                                    info="Minimum OWL-ViT score to pass")
                            clip_slider = gr.Slider(0.01, 0.50, value=0.22, step=0.01,
                                                    label="V2: CLIP Similarity Threshold",
                                                    info="Minimum cosine similarity to pass")
                        run_btn = gr.Button("🔍 Analyze Hallucination", variant="primary", size="lg")

                        gr.Markdown("""
                        > 💡 **Tip:** Try uploading a photo of a dog and prompt
                        > `"a photo of wolf"` to test hard-negative hallucination.
                        """)

                    # Right: Outputs
                    with gr.Column(scale=2):
                        gr.Markdown("### 📤 Detection & Verification")
                        output_image = gr.Image(label="🖼️ Detection Result", height=300)
                        output_html = gr.HTML(label="Analysis")
                        with gr.Accordion("🔬 What CLIP Sees (Cropped Region)", open=False):
                            gr.Markdown(
                                "*The cropped bounding-box region that V2 (CLIP) analyzes. "
                                "Blurry crops → unreliable CLIP scores.*"
                            )
                            cropped_image = gr.Image(label="Cropped Detection", height=180)

                run_btn.click(
                    fn=predict,
                    inputs=[input_image, input_prompt, conf_slider, clip_slider],
                    outputs=[output_image, cropped_image, output_html]
                )

            # ─── TAB 2: Evaluation Metrics ────────────────
            with gr.TabItem("📊 Evaluation Metrics"):
                gr.Markdown("### 📈 Model Performance on COCO val2017 (5,000 samples)")
                gr.Markdown("Results from the full pipeline evaluation (Person 4).")

                with gr.Row():
                    roc_path = load_eval_image("roc_curve_comparison.png")
                    if roc_path:
                        gr.Image(value=roc_path, label="ROC Curve: V1 vs V2",
                                 interactive=False, height=400)
                    else:
                        gr.Markdown(
                            "> ⚠️ `roc_curve_comparison.png` not found. "
                            "Run `evaluation/evaluate.py` first to generate plots."
                        )

                with gr.Row():
                    cm1 = load_eval_image("v1_confusion_matrix.png")
                    cm2 = load_eval_image("v2_confusion_matrix.png")
                    if cm1:
                        gr.Image(value=cm1, label="V1: Confusion Matrix",
                                 interactive=False, height=350)
                    else:
                        gr.Markdown("> ⚠️ V1 confusion matrix not found.")
                    if cm2:
                        gr.Image(value=cm2, label="V2: Confusion Matrix",
                                 interactive=False, height=350)
                    else:
                        gr.Markdown("> ⚠️ V2 confusion matrix not found.")

                gr.HTML("""
                <div class="info-card">
                    <h3 style="margin:0 0 10px;">📋 Key Findings</h3>
                    <table style="width:100%;color:#ccc;font-size:0.9rem;">
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                            <th style="padding:8px;text-align:left;">Metric</th>
                            <th style="padding:8px;text-align:center;">V1 (Confidence)</th>
                            <th style="padding:8px;text-align:center;">V2 (CLIP)</th>
                        </tr>
                        <tr><td style="padding:8px;">Accuracy</td>
                            <td style="padding:8px;text-align:center;color:#6bff6b;">0.8576</td>
                            <td style="padding:8px;text-align:center;">0.8066</td></tr>
                        <tr><td style="padding:8px;">F1-Score</td>
                            <td style="padding:8px;text-align:center;color:#6bff6b;">0.8865</td>
                            <td style="padding:8px;text-align:center;">0.8493</td></tr>
                        <tr><td style="padding:8px;">ROC-AUC</td>
                            <td style="padding:8px;text-align:center;color:#6bff6b;">0.9310</td>
                            <td style="padding:8px;text-align:center;">0.8669</td></tr>
                    </table>
                    <p style="color:#999;font-size:0.8rem;margin-top:10px;">
                        V1 outperforms V2 due to CLIP's cropping resolution bottleneck — small or
                        blurry bounding-box crops degrade CLIP's semantic matching capability.</p>
                </div>
                """)

            # ─── TAB 3: Architecture ──────────────────────
            with gr.TabItem("🧩 System Architecture"):
                gr.Markdown("""
                ### 🏗️ Pipeline Overview

                Our hallucination detection system uses a **two-stage verification** approach
                to determine whether an open-vocabulary detection is genuine or hallucinated.

                ---

                #### Stage 1: Object Detection (OWL-ViT)
                The **OWL-ViT** (Vision Transformer for Open-World Localization) model receives
                an image and a free-form text prompt. It outputs bounding boxes with confidence
                scores for the prompted object. Because OWL-ViT is highly responsive to text,
                it may predict objects based purely on the prompt — even when the object is absent.

                #### Stage 2: Dual Verification

                | Verifier | Method | Strength | Weakness |
                |----------|--------|----------|----------|
                | **V1** (Baseline) | Confidence threshold on OWL-ViT score | Simple, fast, robust | Cannot reason about semantics |
                | **V2** (CLIP) | Cosine similarity between cropped image & text embeddings | Understands meaning | Degrades on small/blurry crops |

                #### Decision Logic
                If **either** verifier flags the detection as hallucinated, the system reports
                `HALLUCINATION DETECTED` (conservative / safety-first approach).

                ---

                ### 🔑 Key Insight from This Research

                > *"Adding model complexity (CLIP) does not guarantee better performance
                > if the image pipeline (cropping) introduces resolution bottlenecks.
                > V1's simplicity proved more robust than V2's semantic understanding."*

                ---

                #### Tech Stack
                - **Detection:** OWL-ViT (`google/owlvit-base-patch32`)
                - **Verification:** CLIP (`openai/clip-vit-base-patch32`)
                - **Dataset:** COCO val2017 (5,000 image-prompt pairs)
                - **Evaluation:** scikit-learn (ROC-AUC, F1, Precision, Recall)
                - **Framework:** PyTorch + Hugging Face Transformers
                - **Demo:** Gradio
                """)

        # Footer
        gr.HTML("""
        <div style="text-align:center;margin-top:16px;padding:10px;color:#666;font-size:0.78rem;">
            Built with OWL-ViT &amp; CLIP · Powered by Gradio &amp; HuggingFace Transformers
        </div>
        """)

    demo.launch(share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio Demo — Hallucination Detection")
    parser.add_argument('--share', action='store_true', help="Create public share link")
    parser.add_argument('--data_dir', type=str, default=None, help="Override data directory")
    args = parser.parse_args()
    build_demo(share=args.share, data_dir=args.data_dir)
