"""Streamlit session helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from ai.gemini_client import GeminiClient
from ai.langchain_pipeline import CareerCounselorChain
from services.utils import get_logger, load_environment

logger = get_logger(__name__)


def initialise_session() -> None:
    load_environment()
    if "auth" not in st.session_state:
        st.session_state["auth"] = None
    if "ai_chain" not in st.session_state:
        st.session_state["ai_chain"] = None


def get_current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("auth")


def set_current_user(user_payload: Dict[str, Any]) -> None:
    st.session_state["auth"] = user_payload
    st.session_state.setdefault("auth_refresh_token", user_payload.get("refreshToken"))


def clear_session() -> None:
    st.session_state["auth"] = None
    st.session_state["ai_chain"] = None
    st.session_state.pop("auth_refresh_token", None)


def require_authentication() -> Dict[str, Any]:
    user = get_current_user()
    if not user:
        st.warning("Please log in from the Home page to access this section.")
        st.stop()
    return user


def get_ai_chain() -> CareerCounselorChain:
    if st.session_state.get("ai_chain") is None:
        logger.info("Creating new AI chain for session")
        gemini_client = GeminiClient()
        st.session_state["ai_chain"] = CareerCounselorChain(gemini_client)
    return st.session_state["ai_chain"]
