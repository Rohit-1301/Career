"""Placeholder for Cloud Functions integrations."""

from __future__ import annotations

from typing import Any, Dict

from services.utils import get_logger

logger = get_logger(__name__)


def enqueue_background_task(name: str, payload: Dict[str, Any]) -> None:
    """Stub function demonstrating where Cloud Function triggers would be invoked."""
    logger.info("Cloud Function '%s' would receive payload: %s", name, payload)
