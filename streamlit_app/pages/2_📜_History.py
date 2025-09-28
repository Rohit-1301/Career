"""History page showing past interactions."""

from __future__ import annotations

from typing import List

import streamlit as st

from firebase import db
from services.utils import get_logger

from streamlit_app.components import session as session_components

logger = get_logger(__name__)

session_components.initialise_session()
user = session_components.require_authentication()

st.title("📜 Conversation History")

if st.button("Refresh history", use_container_width=True):
    st.rerun()

try:
    history: List[dict] = db.list_history(user.get("localId"), limit=50)
except Exception as exc:  # pragma: no cover - external dependency
    st.error(f"Unable to load history: {exc}")
    history = []

if not history:
    st.info("No conversation history yet. Ask a question on the dashboard!")
else:
    for item in history:
        with st.expander(item.get("query", "(no question provided)")):
            st.markdown(f"**AI Response:**\n\n{item.get('response', 'No response recorded.')}")
            metadata = item.get("metadata") or {}
            if metadata:
                st.caption(f"Metadata: {metadata}")
