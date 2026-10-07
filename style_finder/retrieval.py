"""Vector retrieval: find the catalog outfit most similar to a query embedding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import EMBEDDING


@dataclass
class Match:
    row: pd.Series
    score: float


def cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and every row of ``matrix``."""
    query = np.asarray(query, dtype=np.float64).ravel()
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = np.finfo(np.float64).eps
    return (matrix @ query) / norms


class CatalogIndex:
    """In-memory index over the catalog's precomputed embeddings."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog
        self._matrix = np.vstack(catalog[EMBEDDING].to_numpy())

    def best_match(self, query: np.ndarray) -> Match:
        scores = cosine_similarities(query, self._matrix)
        idx = int(np.argmax(scores))
        return Match(row=self.catalog.iloc[idx], score=float(scores[idx]))
