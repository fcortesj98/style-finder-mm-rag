"""Post-processing of the raw LLM output into Markdown for the Gradio UI."""

from __future__ import annotations

import logging
import re

from .prompts import EXACT_MATCH_HEADER, SIMILAR_MATCH_HEADER

logger = logging.getLogger(__name__)

TITLE = "# Fashion Analysis"

REFUSAL_PHRASES = (
    "I'm not able to provide",
    "I cannot provide",
    "I apologize, but I cannot",
    "I don't feel comfortable",
    "violated our content policy",
)

_SECTION_HEADINGS = {
    EXACT_MATCH_HEADER: "## Item Details",
    SIMILAR_MATCH_HEADER: "## Similar Items",
}


def _normalize_bullets(text: str) -> str:
    return re.sub(r"^\* ", "- ", text, flags=re.MULTILINE)


def _escape_dollars(text: str) -> str:
    # Gradio's Markdown treats `$...$` as LaTeX, which garbles prices.
    return text.replace("$", "\\$")


def is_refusal(text: str) -> bool:
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def to_markdown(response: str) -> str:
    """Turn the model's answer into display-ready Markdown.

    If the model refused, salvage the item list (which the prompt asked it to
    repeat) so the user still gets the retrieved products.
    """
    if not response:
        logger.warning("Empty response received")
        return f"{TITLE}\n\nNo detailed analysis was generated. Please try another image."

    if is_refusal(response):
        logger.warning("Model refused the request; falling back to item details")
        for marker, heading in _SECTION_HEADINGS.items():
            if marker in response:
                items = response.split(marker, 1)[1].strip()
                return _escape_dollars(
                    f"{TITLE}\n\nHere are the items detected in your image:\n\n"
                    f"{heading}\n\n{_normalize_bullets(items)}"
                )
        return _escape_dollars(response)

    processed = _escape_dollars(response)
    for marker, heading in _SECTION_HEADINGS.items():
        processed = processed.replace(marker, heading)

    if not processed.startswith("#"):
        processed = f"{TITLE}\n\n{processed}"

    return _normalize_bullets(processed)
