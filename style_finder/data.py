"""Loading and querying the fashion catalog (the RAG knowledge base).

Each row of the dataset is one purchasable item. Items that appear in the same
outfit photo share the same ``Image URL`` and the same precomputed ``Embedding``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column names used in swift-style-embeddings.pkl
ITEM_NAME = "Item Name"
PRICE = "Price"
LINK = "Link"
IMAGE_URL = "Image URL"
EMBEDDING = "Embedding"

REQUIRED_COLUMNS = (ITEM_NAME, PRICE, LINK, IMAGE_URL, EMBEDDING)


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load the pickled catalog and check it has what the pipeline needs.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the dataset is empty or missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download swift-style-embeddings.pkl "
            "(see README) or set STYLE_FINDER_DATASET."
        )

    catalog = pd.read_pickle(path)
    if catalog.empty:
        raise ValueError(f"Dataset at '{path}' is empty")

    missing = [col for col in REQUIRED_COLUMNS if col not in catalog.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Rows without an embedding can never be retrieved, so drop them once here.
    catalog = catalog.dropna(subset=[EMBEDDING]).reset_index(drop=True)
    logger.info("Loaded %d catalog items from %s", len(catalog), path)
    return catalog


def items_for_outfit(catalog: pd.DataFrame, image_url: str) -> pd.DataFrame:
    """Return every catalog item that belongs to the outfit photo ``image_url``."""
    items = catalog[catalog[IMAGE_URL] == image_url]
    logger.info("Found %d items for outfit %s", len(items), image_url)
    return items
