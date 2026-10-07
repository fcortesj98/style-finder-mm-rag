"""Thin client for the Llama 4 vision-instruct model served by IBM watsonx.ai."""

from __future__ import annotations

import logging

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from .config import WatsonxSettings

logger = logging.getLogger(__name__)


class VisionLLM:
    """Sends an image plus a text prompt to a watsonx.ai chat model."""

    def __init__(self, settings: WatsonxSettings | None = None):
        settings = settings or WatsonxSettings()
        credentials = Credentials(url=settings.url, api_key=settings.api_key)
        self.model = ModelInference(
            model_id=settings.model_id,
            credentials=credentials,
            project_id=settings.project_id,
            params=TextChatParameters(
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
            ),
        )

    def chat_with_image(self, prompt: str, image_base64: str) -> str:
        """Return the model's reply, or an error message string if the call fails."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        try:
            logger.info("Sending request to LLM (prompt length %d)", len(prompt))
            response = self.model.chat(messages=messages)
        except Exception as exc:  # network/auth/quota errors from the SDK
            logger.exception("LLM request failed")
            return f"Error generating response: {exc}"

        choice = response["choices"][0]
        content = choice["message"]["content"]
        logger.info("Received response (length %d)", len(content))
        if choice.get("finish_reason") == "length":
            logger.warning("Response was truncated at max_tokens")
        return content
