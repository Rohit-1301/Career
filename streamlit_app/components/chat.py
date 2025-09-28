"""Chat UI helpers for Streamlit."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import streamlit as st

from services.utils import get_logger

logger = get_logger(__name__)


def render_chat_history(history: Iterable[Mapping[str, str]]) -> None:
    for message in history:
        role = message.get("role") or message.get("author") or "user"
        content = message.get("content") or message.get("response")
        if not content:
            continue
        with st.chat_message("assistant" if role in {"assistant", "ai"} else "user"):
            st.markdown(content)


def render_response_message(text: str) -> None:
    with st.chat_message("assistant"):
        st.markdown(text)


def render_user_message(text: str) -> None:
    with st.chat_message("user"):
        st.markdown(text)


def render_empty_state() -> None:
    st.info("Ask your first career question to get started!")
