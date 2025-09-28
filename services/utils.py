"""Utility helpers shared across the CareerSaathi application."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_environment(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from a `.env` file once per process."""
    search_paths = []
    if dotenv_path:
        search_paths.append(Path(dotenv_path))
    # Search from this file up to project root for convenience.
    search_paths.extend(
        parent / ".env"
        for parent in Path(__file__).resolve().parents
        if parent.name != ""
    )

    for candidate in search_paths:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break
    else:
        # Fallback to default behaviour (load_dotenv with no path)
        load_dotenv(override=False)


def get_env_variable(key: str, default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    """Fetch an environment variable with optional fallback and validation."""
    load_environment()
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Environment variable '{key}' is required but missing.")
    return value


@lru_cache(maxsize=None)
def get_logger(name: str = "careersaathi") -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into evenly sized chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
