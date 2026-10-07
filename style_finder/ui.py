"""Gradio user interface."""

from __future__ import annotations

from typing import Callable

import gradio as gr

from .config import EXAMPLES_DIR

INTRO = """
# Fashion Style Analyzer

Upload an outfit photo to identify its garments and get matching catalog items with
prices and links. Under the hood this combines computer vision (ResNet50 embeddings),
vector similarity search and a vision LLM (Llama 4 Maverick on watsonx.ai).
"""

ABOUT = """
### How it works

1. **Encode** – the image is turned into a ResNet50 feature vector.
2. **Retrieve** – cosine similarity finds the closest outfit in the catalog.
3. **Generate** – the retrieved items are added to the prompt, and Llama 4 describes
   the outfit and lists the matching products.
"""


def example_images() -> list[str]:
    return sorted(str(p) for p in EXAMPLES_DIR.glob("*.png"))


def build_interface(analyze: Callable[[object], str]) -> gr.Blocks:
    """Build the Gradio app around an ``analyze(image) -> markdown`` function."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Style Analyzer") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Fashion Image")
                submit_btn = gr.Button("Analyze Style", variant="primary")
                status = gr.Markdown("Ready to analyze.")
                gr.Examples(
                    examples=example_images(),
                    inputs=image_input,
                    label="Example images (click one to load it)",
                    examples_per_page=6,
                )

            with gr.Column(scale=2):
                output = gr.Markdown(label="Style Analysis Results", height=700)

        gr.Markdown(ABOUT)

        submit_btn.click(
            fn=lambda: "Analyzing image... This may take a few moments.",
            outputs=status,
        ).then(
            fn=analyze,
            inputs=image_input,
            outputs=output,
        ).then(
            fn=lambda: "Analysis complete!",
            outputs=status,
        )

    return demo
