"""Entry point: `python app.py` starts the Style Finder web app on port 5000."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from style_finder.config import Settings
from style_finder.pipeline import StyleFinder
from style_finder.ui import build_interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Style Finder Gradio app.")
    parser.add_argument("--dataset", type=Path, help="Path to swift-style-embeddings.pkl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public gradio.live link"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    args = parse_args()

    settings = Settings.from_env()
    if args.dataset:
        settings = replace(settings, dataset_path=args.dataset)

    finder = StyleFinder(settings)
    demo = build_interface(finder.analyze)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
