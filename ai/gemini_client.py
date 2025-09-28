"""Google Gemini client wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from services.utils import get_env_variable, get_logger

logger = get_logger(__name__)


@dataclass
class GeminiClient:
    """Lightweight wrapper for configuring Gemini models."""

    model_name: str = "gemini-2.0-flash-lite"
    temperature: float = 0.4
    top_p: float = 0.8
    top_k: int = 40

    def __post_init__(self) -> None:
        api_key = get_env_variable("GEMINI_API_KEY", required=True)
        genai.configure(api_key=api_key)
        logger.info("Configured Gemini client for model %s", self.model_name)

    def get_llm(self, *, temperature: Optional[float] = None) -> ChatGoogleGenerativeAI:
        """Return a LangChain-compatible chat model."""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

    def generate_text(self, prompt: str, *, temperature: Optional[float] = None) -> str:
        """Generate free-form text using the underlying Gemini SDK."""
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature if temperature is not None else self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )
        parts = [part.text for part in response.candidates[0].content.parts if getattr(part, "text", None)]
        return "".join(parts)
