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
        Returns annotated image and result text.
        """
        if image is None or not prompt.strip():
            return None, "Please upload an image and enter a prompt."

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
            return image, "No detection at all. The model found nothing."

        # Get best detection
        best_idx = torch.argmax(results["scores"]).item()
        best_score = results["scores"][best_idx].item()
        best_box = results["boxes"][best_idx].tolist()

        # --- Step 2: V1 — Confidence Threshold ---
        v1_result = "GENUINE" if best_score >= conf_threshold else "HALLUCINATION"

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

        v2_result = "GENUINE" if clip_sim >= clip_threshold else "HALLUCINATION"

        # --- Step 4: Draw bounding box on image ---
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Color based on final verdict
        is_hallucination = (v1_result == "HALLUCINATION") or (v2_result == "HALLUCINATION")
        box_color = "#F44336" if is_hallucination else "#4CAF50"  # Red or Green

        draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=3)

        # Add label
        label_text = f"{prompt} ({best_score:.2f})"
        draw.text((xmin + 5, ymin - 15), label_text, fill=box_color)

        # --- Step 5: Build result text ---
        verdict = "HALLUCINATION DETECTED" if is_hallucination else "GENUINE DETECTION"
        emoji = "WARNING" if is_hallucination else "OK"

        result_text = f"""
========================================
  [{emoji}] {verdict}
========================================

Prompt: "{prompt}"

--- V1: Confidence Threshold ---
  OWL-ViT Score : {best_score:.4f}
  Threshold     : {conf_threshold}
  Verdict       : {v1_result}

--- V2: CLIP Similarity ---
  CLIP Score    : {clip_sim:.4f}
  Threshold     : {clip_threshold}
  Verdict       : {v2_result}

--- Bounding Box ---
  Coordinates   : [{xmin}, {ymin}, {xmax}, {ymax}]
"""
        return annotated, result_text


def build_demo(share=False):
    """Build and launch the Gradio demo interface."""

    print("Initializing models (this may take a minute)...")
    detector = HallucinationDetector()

    def predict(image, prompt, conf_threshold, clip_threshold):
        return detector.detect_and_verify(image, prompt, conf_threshold, clip_threshold)

    # Build Gradio UI
    with gr.Blocks(
        title="Hallucination Detection Demo",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # Prompt Hallucination Detection
        ### Detecting false object detections in OWL-ViT caused by misleading text prompts

        Upload an image and enter a text prompt. The system will:
        1. Run **OWL-ViT** to detect the prompted object
        2. Verify using **V1 (Confidence Threshold)**
        3. Verify using **V2 (CLIP Similarity)**
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                input_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. a photo of wolf")

                with gr.Row():
                    conf_slider = gr.Slider(0.01, 0.99, value=0.15, step=0.01,
                                           label="V1: Confidence Threshold")
                    clip_slider = gr.Slider(0.01, 0.50, value=0.22, step=0.01,
                                           label="V2: CLIP Similarity Threshold")

                run_btn = gr.Button("Detect Hallucination", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Detection Result")
                output_text = gr.Textbox(label="Analysis", lines=18)

        run_btn.click(
            fn=predict,
            inputs=[input_image, input_prompt, conf_slider, clip_slider],
            outputs=[output_image, output_text]
        )

        gr.Markdown("""
        ---
        **How it works:**
        - **Green box** = Genuine detection (the object likely exists)
        - **Red box** = Hallucination detected (the model was misled by the prompt)
        - **V1** checks if OWL-ViT's confidence score is above a threshold
        - **V2** uses CLIP to verify if the detected region actually matches the prompt
        """)

    demo.launch(share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio Demo for Hallucination Detection")
    parser.add_argument('--share', action='store_true', help="Create public share link")
    parser.add_argument('--data_dir', type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    build_demo(share=args.share)
