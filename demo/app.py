"""
app.py — Person 4 (Evaluation & Demo)

Interactive Gradio Web Demo for Hallucination Detection.
Users can upload an image + type a prompt → see if OWL-ViT hallucinates.

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
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    CLIPProcessor, CLIPModel
)


# ---------------------------------------------------------------------------
# Custom CSS for a premium dark-themed UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global tweaks ────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

/* ── Hero header ──────────────────────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-section h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 4px !important;
}
.hero-section p, .hero-section li {
    color: #b0b3c5 !important;
    font-size: 0.95rem;
}

/* ── Verdict cards ────────────────────────────────────── */
.verdict-hallucination {
    background: linear-gradient(135deg, #1a0000 0%, #330000 100%);
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 24px;
    animation: pulse-red 2s ease-in-out infinite;
}
.verdict-genuine {
    background: linear-gradient(135deg, #001a00 0%, #003300 100%);
    border: 2px solid #44ff44;
    border-radius: 16px;
    padding: 24px;
    animation: pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.3); }
    50% { box-shadow: 0 0 40px rgba(255, 68, 68, 0.6); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px rgba(68, 255, 68, 0.3); }
    50% { box-shadow: 0 0 40px rgba(68, 255, 68, 0.6); }
}

/* ── Score bars ───────────────────────────────────────── */
.score-container {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.08);
}
.score-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 24px;
    overflow: hidden;
    margin-top: 6px;
}
.score-bar-fill-red {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #ff4444, #ff6b6b);
    transition: width 0.8s ease;
}
.score-bar-fill-green {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #44ff44, #6bff6b);
    transition: width 0.8s ease;
}
.score-bar-fill-amber {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #ffaa00, #ffd000);
    transition: width 0.8s ease;
}

/* ── How-it-works footer ──────────────────────────────── */
.info-card {
    background: rgba(102, 126, 234, 0.08);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
}

/* ── Slider labels ────────────────────────────────────── */
.threshold-section label {
    font-weight: 600 !important;
}
"""


class HallucinationDetector:
    """Combined detector + verifier for the demo."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load OWL-ViT
        print("Loading OWL-ViT...")
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.owl_model.eval()

        # Load CLIP
        print("Loading CLIP...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()

        print("Models loaded!")

    def detect_and_verify(self, image, prompt, conf_threshold=0.15, clip_threshold=0.22):
        """
        Run the full pipeline: detect → verify → annotate.
        Returns annotated image, cropped image, and result HTML.
        """
        if image is None or not prompt.strip():
            return None, None, "<p style='color:#ff6b6b;'>⚠️ Please upload an image and enter a text prompt.</p>"

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # --- Step 1: OWL-ViT Detection ---
        texts = [[prompt]]
        inputs = self.owl_processor(text=texts, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.owl_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owl_processor.image_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

        if len(results["scores"]) == 0:
            return image, None, "<p style='color:#ffd000;'>🤷 No detection at all — the model found nothing in this image.</p>"

        # Get best detection
        best_idx = torch.argmax(results["scores"]).item()
        best_score = results["scores"][best_idx].item()
        best_box = results["boxes"][best_idx].tolist()

        # --- Step 2: V1 — Confidence Threshold ---
        v1_pass = best_score >= conf_threshold
        v1_result = "GENUINE" if v1_pass else "HALLUCINATION"

        # --- Step 3: V2 — CLIP Similarity ---
        xmin, ymin, xmax, ymax = [int(c) for c in best_box]
        w, h = image.size
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        if xmax > xmin and ymax > ymin:
            cropped = image.crop((xmin, ymin, xmax, ymax))
        else:
            cropped = image

        clip_inputs = self.clip_processor(text=[prompt], images=cropped, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            clip_outputs = self.clip_model(**clip_inputs)
            img_emb = clip_outputs.image_embeds / clip_outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = clip_outputs.text_embeds / clip_outputs.text_embeds.norm(dim=-1, keepdim=True)
            clip_sim = (img_emb @ txt_emb.T).item()

        v2_pass = clip_sim >= clip_threshold
        v2_result = "GENUINE" if v2_pass else "HALLUCINATION"

        # --- Step 4: Draw bounding box on image ---
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        is_hallucination = (not v1_pass) or (not v2_pass)
        box_color = "#FF4444" if is_hallucination else "#44FF44"

        # Draw thicker bounding box
        for i in range(4):
            draw.rectangle(
                [xmin - i, ymin - i, xmax + i, ymax + i],
                outline=box_color
            )

        # Add label with background
        label_text = f"{prompt} ({best_score:.2f})"
        text_bbox = draw.textbbox((xmin, ymin - 20), label_text)
        draw.rectangle(
            [text_bbox[0] - 4, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=box_color
        )
        draw.text((xmin, ymin - 20), label_text, fill="white")

        # --- Step 5: Build rich HTML result ---
        result_html = self._build_result_html(
            prompt, best_score, conf_threshold, v1_pass, v1_result,
            clip_sim, clip_threshold, v2_pass, v2_result,
            is_hallucination, xmin, ymin, xmax, ymax
        )

        return annotated, cropped, result_html

    def _build_result_html(self, prompt, owl_score, conf_thresh, v1_pass, v1_result,
                           clip_score, clip_thresh, v2_pass, v2_result,
                           is_hallucination, xmin, ymin, xmax, ymax):
        """Build a visually rich HTML card for the analysis output."""

        # Final verdict
        if is_hallucination:
            verdict_class = "verdict-hallucination"
            verdict_emoji = "🚨"
            verdict_text = "HALLUCINATION DETECTED"
            verdict_desc = "The model was likely misled by the text prompt. The detected region does not reliably match the description."
        else:
            verdict_class = "verdict-genuine"
            verdict_emoji = "✅"
            verdict_text = "GENUINE DETECTION"
            verdict_desc = "Both verification methods agree — the detected object matches the prompt with sufficient confidence."

        # Score bar helpers
        def score_bar(score, threshold, passed):
            pct = min(score * 100, 100)
            thresh_pct = min(threshold * 100, 100)
            if passed:
                bar_class = "score-bar-fill-green"
            elif score > threshold * 0.7:
                bar_class = "score-bar-fill-amber"
            else:
                bar_class = "score-bar-fill-red"

            status = "✅ PASS" if passed else "❌ FAIL"
            status_color = "#44ff44" if passed else "#ff4444"

            return f"""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.5rem; font-weight:700;">{score:.4f}</span>
                <span style="color:{status_color}; font-weight:700; font-size:1rem;">{status}</span>
            </div>
            <div class="score-bar-bg" style="position:relative;">
                <div class="{bar_class}" style="width:{pct}%;"></div>
                <div style="position:absolute; left:{thresh_pct}%; top:0; height:100%;
                     border-left:2px dashed rgba(255,255,255,0.7);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px;">
                <span style="color:#888; font-size:0.8rem;">0.0</span>
                <span style="color:#aaa; font-size:0.8rem;">▲ Threshold: {threshold:.2f}</span>
                <span style="color:#888; font-size:0.8rem;">1.0</span>
            </div>
            """

        html = f"""
        <div class="{verdict_class}">
            <div style="text-align:center; margin-bottom:12px;">
                <span style="font-size:3rem;">{verdict_emoji}</span>
                <h2 style="margin:8px 0 4px 0; font-size:1.6rem;
                    color:{'#ff6b6b' if is_hallucination else '#6bff6b'};">
                    {verdict_text}
                </h2>
                <p style="color:#aaa; margin:0; font-size:0.9rem;">{verdict_desc}</p>
            </div>
        </div>

        <div style="margin-top:16px;">
            <p style="color:#ccc; font-size:0.9rem;">
                🔍 Prompt: <strong style="color:#fff;">"{prompt}"</strong>
            </p>
        </div>

        <div class="score-container">
            <h3 style="margin:0 0 8px 0; font-size:1rem;">
                🎯 V1: OWL-ViT Confidence Score
            </h3>
            <p style="color:#999; font-size:0.8rem; margin:0 0 8px 0;">
                How confident is OWL-ViT that this object exists?
            </p>
            {score_bar(owl_score, conf_thresh, v1_pass)}
        </div>

        <div class="score-container">
            <h3 style="margin:0 0 8px 0; font-size:1rem;">
                🧠 V2: CLIP Semantic Similarity
            </h3>
            <p style="color:#999; font-size:0.8rem; margin:0 0 8px 0;">
                Does the cropped region actually look like the prompt? (See "What CLIP Sees" below)
            </p>
            {score_bar(clip_score, clip_thresh, v2_pass)}
        </div>

        <div style="background:rgba(255,255,255,0.03); border-radius:8px;
             padding:12px 16px; margin-top:8px; border:1px solid rgba(255,255,255,0.06);">
            <p style="color:#888; font-size:0.8rem; margin:0;">
                📐 Bounding Box: [{xmin}, {ymin}, {xmax}, {ymax}]
            </p>
        </div>
        """
        return html


def build_demo(share=False):
    """Build and launch the Gradio demo interface."""

    print("Initializing models (this may take a minute)...")
    detector = HallucinationDetector()

    def predict(image, prompt, conf_threshold, clip_threshold):
        return detector.detect_and_verify(image, prompt, conf_threshold, clip_threshold)

    # Build Gradio UI
    with gr.Blocks(
        title="Hallucination Detection Demo",
        theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"),
        css=CUSTOM_CSS
    ) as demo:

        # ── Hero Header ──
        gr.HTML("""
        <div class="hero-section">
            <h1>🔍 Prompt Hallucination Detection</h1>
            <p style="font-size:1.05rem; color:#d0d3e5 !important; margin-top:8px;">
                Does your AI <em>really</em> see what you asked for — or is it just making things up?
            </p>
            <p style="margin-top:12px;">
                Upload any image and type a text prompt. The system runs two independent checks:
            </p>
            <ol style="margin-top:6px; padding-left:20px;">
                <li><strong style="color:#667eea;">OWL-ViT</strong> scans the image for the object you described</li>
                <li><strong style="color:#ffa726;">V1</strong> checks OWL-ViT's confidence score against a threshold</li>
                <li><strong style="color:#ab47bc;">V2 (CLIP)</strong> crops the detected region and verifies it semantically</li>
            </ol>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Left Column: Inputs ──
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📷 Upload Image",
                    type="pil",
                    height=320
                )
                input_prompt = gr.Textbox(
                    label="✏️ Text Prompt",
                    placeholder='e.g. "a photo of wolf" or "ships on the river"',
                    info="Describe the object you want the model to find."
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    gr.Markdown(
                        "*Adjust the decision thresholds. "
                        "Higher values = stricter verification (fewer false positives).*"
                    )
                    conf_slider = gr.Slider(
                        0.01, 0.99, value=0.15, step=0.01,
                        label="V1: Confidence Threshold",
                        info="Minimum OWL-ViT score to pass"
                    )
                    clip_slider = gr.Slider(
                        0.01, 0.50, value=0.22, step=0.01,
                        label="V2: CLIP Similarity Threshold",
                        info="Minimum CLIP cosine similarity to pass"
                    )

                run_btn = gr.Button(
                    "🚀 Detect Hallucination",
                    variant="primary",
                    size="lg"
                )

            # ── Right Column: Outputs ──
            with gr.Column(scale=1):
                output_image = gr.Image(label="🖼️ Detection Result", height=320)
                output_html = gr.HTML(label="Analysis")

                with gr.Accordion("🔬 What CLIP Sees (Cropped Region)", open=False):
                    gr.Markdown(
                        "*This is the cropped bounding-box region that V2 (CLIP) actually analyzes. "
                        "If this image is blurry or too small, CLIP may give unreliable scores.*"
                    )
                    cropped_image = gr.Image(label="Cropped Detection", height=200)

        # ── Wire up the button ──
        run_btn.click(
            fn=predict,
            inputs=[input_image, input_prompt, conf_slider, clip_slider],
            outputs=[output_image, cropped_image, output_html]
        )

        # ── Example gallery ──
        gr.Markdown("### 💡 Try These Examples")
        gr.Markdown(
            "*Click any row below to auto-fill the inputs, then press **Detect Hallucination**.*"
        )

        # ── Info footer ──
        gr.HTML("""
        <div class="info-card">
            <h3 style="margin:0 0 12px 0;">ℹ️ How to Read the Results</h3>
            <table style="width:100%; color:#ccc; font-size:0.9rem;">
                <tr>
                    <td style="padding:6px 12px;">🟢 <strong>Green Box</strong></td>
                    <td style="padding:6px 12px;">Genuine — Both V1 and V2 agree the object is real</td>
                </tr>
                <tr>
                    <td style="padding:6px 12px;">🔴 <strong>Red Box</strong></td>
                    <td style="padding:6px 12px;">Hallucination — At least one verifier flagged a mismatch</td>
                </tr>
                <tr>
                    <td style="padding:6px 12px;">📊 <strong>Score Bars</strong></td>
                    <td style="padding:6px 12px;">Visual gauge showing how close the score is to the threshold (dashed line)</td>
                </tr>
                <tr>
                    <td style="padding:6px 12px;">🔬 <strong>Cropped View</strong></td>
                    <td style="padding:6px 12px;">The exact image region that CLIP analyzes — if it's blurry, expect low scores</td>
                </tr>
            </table>
        </div>
        """)

        gr.HTML("""
        <div style="text-align:center; margin-top:20px; padding:12px; color:#666; font-size:0.8rem;">
            Built with OWL-ViT (Google) &amp; CLIP (OpenAI) · Powered by Gradio &amp; HuggingFace Transformers
        </div>
        """)

    demo.launch(share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio Demo for Hallucination Detection")
    parser.add_argument('--share', action='store_true', help="Create public share link")
    parser.add_argument('--data_dir', type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    build_demo(share=args.share)
