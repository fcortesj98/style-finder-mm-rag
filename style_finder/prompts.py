"""Prompt construction: augments the LLM request with the retrieved catalog items."""

from __future__ import annotations

import pandas as pd

from .data import ITEM_NAME, LINK, PRICE

EXACT_MATCH_HEADER = "ITEM DETAILS:"
SIMILAR_MATCH_HEADER = "SIMILAR ITEMS:"

_PREAMBLE = (
    "You're conducting a professional retail catalog analysis. "
    "This image shows standard clothing items available in department stores. "
    "Focus exclusively on professional fashion analysis for a clothing retailer. "
)

_EXACT_INSTRUCTIONS = (
    "Please:\n"
    "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
    "2. Categorize the overall style (business, casual, etc.)\n"
    "3. Include the ITEM DETAILS section at the end\n\n"
)

_SIMILAR_INSTRUCTIONS = (
    "Please:\n"
    "1. Note these are similar but not exact items\n"
    "2. Identify clothing elements objectively (colors, patterns, materials)\n"
    "3. Include the SIMILAR ITEMS section at the end\n\n"
)

_CLOSING = "This is for a professional retail catalog. Use formal, clinical language."

# Below this length the model's answer is treated as a failed/empty generation.
MIN_USEFUL_RESPONSE_LENGTH = 100


def section_header(is_exact_match: bool) -> str:
    return EXACT_MATCH_HEADER if is_exact_match else SIMILAR_MATCH_HEADER


def format_items(items: pd.DataFrame) -> str:
    """Render retrieved items as a Markdown bullet list with price and link."""
    return "\n".join(
        f"- {row[ITEM_NAME]} (${row[PRICE]}): {row[LINK]}" for _, row in items.iterrows()
    )


def build_fashion_prompt(items_markdown: str, is_exact_match: bool) -> str:
    """Build the retrieval-augmented prompt sent alongside the user's image."""
    header = section_header(is_exact_match)
    label = header.rstrip(":")
    instructions = _EXACT_INSTRUCTIONS if is_exact_match else _SIMILAR_INSTRUCTIONS
    return (
        f"{_PREAMBLE}"
        f"{label} (always include this section in your response):\n{items_markdown}\n\n"
        f"{instructions}"
        f"{_CLOSING}"
    )


def ensure_items_section(response: str, items_markdown: str, is_exact_match: bool) -> str:
    """Guarantee the retrieved items reach the user even if the model drops them."""
    header = section_header(is_exact_match)

    if len(response) < MIN_USEFUL_RESPONSE_LENGTH:
        return (
            "# Fashion Analysis\n\n"
            "This outfit features a collection of carefully coordinated pieces.\n\n"
            f"{header}\n{items_markdown}"
        )

    if EXACT_MATCH_HEADER not in response and SIMILAR_MATCH_HEADER not in response:
        return f"{response}\n\n{header}\n{items_markdown}"

    return response
