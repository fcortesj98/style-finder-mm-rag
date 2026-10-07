"""The multimodal RAG pipeline: encode -> retrieve -> augment -> generate."""

from __future__ import annotations

import logging

from . import data, formatting, prompts
from .config import Settings
from .encoder import ImageEncoder, ImageInput
from .llm import VisionLLM
from .retrieval import CatalogIndex

logger = logging.getLogger(__name__)


class StyleFinder:
    """Analyzes an outfit photo using the catalog as retrieval context."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.catalog = data.load_catalog(self.settings.dataset_path)
        self.index = CatalogIndex(self.catalog)
        self.encoder = ImageEncoder(self.settings.encoder)
        self.llm = VisionLLM(self.settings.watsonx)

    def analyze(self, image: ImageInput | None) -> str:
        """Run the full pipeline and return Markdown for display."""
        if image is None:
            return "Please upload an image or pick an example first."

        # 1. Encode the query image into a vector (+ Base64 for the LLM).
        try:
            encoded = self.encoder.encode(image)
        except Exception:
            logger.exception("Failed to encode image")
            return "Error: Unable to process the image. Please try another image."

        # 2. Retrieve the most similar outfit from the catalog.
        match = self.index.best_match(encoded.vector)
        logger.info(
            "Closest match: %s (similarity %.2f)", match.row[data.ITEM_NAME], match.score
        )

        # 3. Collect every purchasable item in that outfit.
        items = data.items_for_outfit(self.catalog, match.row[data.IMAGE_URL])
        if items.empty:
            return "Error: No items found for the matched image."

        # 4. Augment the prompt with the retrieved items and generate.
        is_exact = match.score >= self.settings.similarity_threshold
        items_md = prompts.format_items(items)
        prompt = prompts.build_fashion_prompt(items_md, is_exact)
        raw = self.llm.chat_with_image(prompt, encoded.base64_jpeg)

        response = prompts.ensure_items_section(raw, items_md, is_exact)
        return formatting.to_markdown(response)
