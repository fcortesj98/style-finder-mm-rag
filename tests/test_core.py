"""Unit tests for the parts of the pipeline that don't need torch or watsonx.ai."""

import numpy as np
import pandas as pd
import pytest

from style_finder import data, formatting, prompts
from style_finder.retrieval import CatalogIndex, cosine_similarities


@pytest.fixture
def catalog():
    return pd.DataFrame(
        {
            data.ITEM_NAME: ["Red Blazer", "Black Skirt", "Blue Top"],
            data.PRICE: [120, 80, 40],
            data.LINK: ["https://a", "https://b", "https://c"],
            data.IMAGE_URL: ["img1", "img1", "img2"],
            data.EMBEDDING: [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )


def test_load_catalog_validates(tmp_path, catalog):
    with pytest.raises(FileNotFoundError):
        data.load_catalog(tmp_path / "missing.pkl")

    bad = tmp_path / "bad.pkl"
    catalog.drop(columns=[data.LINK]).to_pickle(bad)
    with pytest.raises(ValueError, match="missing required columns"):
        data.load_catalog(bad)


def test_load_catalog_drops_rows_without_embedding(tmp_path, catalog):
    catalog.loc[1, data.EMBEDDING] = None
    path = tmp_path / "catalog.pkl"
    catalog.to_pickle(path)
    loaded = data.load_catalog(path)
    assert list(loaded[data.ITEM_NAME]) == ["Red Blazer", "Blue Top"]


def test_cosine_similarities():
    scores = cosine_similarities(np.array([1.0, 0.0]), np.array([[2.0, 0.0], [0.0, 3.0]]))
    np.testing.assert_allclose(scores, [1.0, 0.0])


def test_best_match_and_outfit_items(catalog):
    match = CatalogIndex(catalog).best_match(np.array([0.1, 0.9]))
    assert match.row[data.ITEM_NAME] == "Blue Top"
    assert match.score == pytest.approx(0.9939, abs=1e-3)

    items = data.items_for_outfit(catalog, "img1")
    assert list(items[data.ITEM_NAME]) == ["Red Blazer", "Black Skirt"]


def test_prompt_contains_items_and_right_section(catalog):
    items_md = prompts.format_items(catalog.iloc[:1])
    assert items_md == "- Red Blazer ($120): https://a"
    assert "ITEM DETAILS (always include" in prompts.build_fashion_prompt(items_md, True)
    assert "SIMILAR ITEMS (always include" in prompts.build_fashion_prompt(items_md, False)


def test_ensure_items_section():
    items_md = "- Red Blazer ($120): https://a"
    short = prompts.ensure_items_section("too short", items_md, True)
    assert "ITEM DETAILS:" in short and items_md in short

    long_reply = "A detailed description of the outfit. " * 5
    appended = prompts.ensure_items_section(long_reply, items_md, False)
    assert appended.endswith(f"SIMILAR ITEMS:\n{items_md}")

    already = long_reply + "\nITEM DETAILS:\n" + items_md
    assert prompts.ensure_items_section(already, items_md, True) == already


def test_to_markdown_formats_sections_and_prices():
    md = formatting.to_markdown("Nice outfit.\n\nITEM DETAILS:\n* Red Blazer ($120)")
    assert md.startswith("# Fashion Analysis")
    assert "## Item Details" in md
    assert "- Red Blazer (\\$120)" in md


def test_to_markdown_salvages_items_from_refusal():
    md = formatting.to_markdown("I cannot provide that.\nSIMILAR ITEMS:\n* Blue Top ($40)")
    assert "## Similar Items" in md
    assert "- Blue Top (\\$40)" in md
    assert "I cannot provide" not in md


def test_to_markdown_empty():
    assert "No detailed analysis" in formatting.to_markdown("")
