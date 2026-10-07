"""Image encoding: turns an image into a ResNet50 feature vector and a Base64 string.

The vector is used for similarity search against the catalog; the Base64
string is what gets sent to the vision LLM.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from .config import EncoderSettings

logger = logging.getLogger(__name__)

ImageInput = Image.Image | str | Path


@dataclass
class EncodedImage:
    vector: np.ndarray
    base64_jpeg: str


class ImageEncoder:
    """Wraps a pre-trained ResNet50 used as a fixed feature extractor."""

    def __init__(self, settings: EncoderSettings | None = None):
        settings = settings or EncoderSettings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # IMAGENET1K_V1 is what the legacy `pretrained=True` flag loaded, and it
        # is what the dataset embeddings were built with. Do not switch to
        # DEFAULT (V2) or the stored vectors will no longer be comparable.
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(settings.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=settings.norm_mean, std=settings.norm_std),
            ]
        )

    def encode(self, image: ImageInput) -> EncodedImage:
        """Encode a PIL image, a local file path, or an http(s) URL."""
        pil_image = load_image(image)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return EncodedImage(
            vector=features.cpu().numpy().flatten(),
            base64_jpeg=to_base64_jpeg(pil_image),
        )


def load_image(image: ImageInput) -> Image.Image:
    """Return an RGB PIL image from a PIL image, a file path, or a URL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    source = str(image)
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(source).convert("RGB")


def to_base64_jpeg(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
