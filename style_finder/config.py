"""Central configuration for Style Finder.

Every value has a sensible default for the IBM Skills Network lab environment
and can be overridden with an environment variable, so the app also runs
against your own watsonx.ai account without touching the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class WatsonxSettings:
    """Connection and generation settings for the watsonx.ai chat model."""

    model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    url: str = "https://us-south.ml.cloud.ibm.com"
    # The lab environment injects credentials, so no key is needed there.
    api_key: str | None = None
    project_id: str = "skills-network"
    temperature: float = 0.2
    top_p: float = 0.6
    max_tokens: int = 2000

    @classmethod
    def from_env(cls) -> "WatsonxSettings":
        return cls(
            model_id=_env("WATSONX_MODEL_ID", cls.model_id),
            url=_env("WATSONX_URL", cls.url),
            api_key=_env("WATSONX_APIKEY"),
            project_id=_env("WATSONX_PROJECT_ID", cls.project_id),
        )


@dataclass(frozen=True)
class EncoderSettings:
    """Preprocessing used by the ResNet50 image encoder.

    These must match the settings used to build the dataset embeddings,
    otherwise similarity scores become meaningless.
    """

    image_size: tuple[int, int] = (224, 224)
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Settings:
    """Top-level application settings."""

    dataset_path: Path = PROJECT_ROOT / "swift-style-embeddings.pkl"
    # Cosine similarity at or above this value is treated as an exact outfit match.
    similarity_threshold: float = 0.8
    watsonx: WatsonxSettings = field(default_factory=WatsonxSettings)
    encoder: EncoderSettings = field(default_factory=EncoderSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dataset_path=Path(_env("STYLE_FINDER_DATASET", str(cls.dataset_path))),
            similarity_threshold=float(
                _env("STYLE_FINDER_SIMILARITY_THRESHOLD", str(cls.similarity_threshold))
            ),
            watsonx=WatsonxSettings.from_env(),
        )
