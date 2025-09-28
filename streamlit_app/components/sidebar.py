"""Sidebar layout helpers."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from services.utils import get_logger

logger = get_logger(__name__)


def render_authenticated_sidebar(user: Dict[str, Any]) -> bool:
    st.sidebar.title("CareerSaathi")
    st.sidebar.success(f"Signed in as {user.get('email', 'guest')}")
    with st.sidebar.expander("Session details"):
        st.write(
            {
                "uid": user.get("localId"),
                "emailVerified": user.get("emailVerified"),
            }
        )
    return st.sidebar.button("Sign out", use_container_width=True)
