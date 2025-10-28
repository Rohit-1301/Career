"""Interactive dashboard for CareerSaathi."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from firebase import db, storage
from services.utils import get_logger

from streamlit_app.components import chat as chat_components
from streamlit_app.components import session as session_components

logger = get_logger(__name__)

session_components.initialise_session()
user = session_components.require_authentication()
chain = session_components.get_ai_chain()

st.title("🏠 Chat")

if "uploaded_file_markers" not in st.session_state:
    st.session_state["uploaded_file_markers"] = []
if "latest_upload_context" not in st.session_state:
    st.session_state["latest_upload_context"] = None

# Check if Firebase Storage is available
try:
    from firebase.config import get_storage_bucket
    get_storage_bucket()  # Test if bucket is accessible
    storage_available = True
except Exception as e:
    logger.warning(f"Firebase Storage not available: {e}")
    storage_available = False

with st.container():
    if storage_available:
        uploaded_file = st.file_uploader(
            "Upload supporting documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            marker = f"{uploaded_file.name}-{uploaded_file.size}"
            if marker not in st.session_state["uploaded_file_markers"]:
                try:
                    upload_info = storage.upload_user_file(
                        user.get("localId"),
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        content_type=uploaded_file.type,
                    )
                    db.record_file_reference(
                        user.get("localId"),
                        upload_info["path"],
                        metadata={"contentType": uploaded_file.type},
                    )
                    st.session_state["uploaded_file_markers"].append(marker)
                    st.success(f"Uploaded {uploaded_file.name}")

                    if uploaded_file.type in {"text/plain", "application/json"} and uploaded_file.size < 1_000_000:
                        try:
                            st.session_state["latest_upload_context"] = uploaded_file.getvalue().decode("utf-8")
                            st.info("We'll include the uploaded text as context for your next question.")
                        except UnicodeDecodeError:
                            st.warning("Could not decode the uploaded file as UTF-8 text. It will be stored but not analysed automatically.")
                    else:
                        st.session_state["latest_upload_context"] = None
                except Exception as exc:  # pragma: no cover - external service interaction
                    st.error(f"Failed to upload file: {exc}")
    else:
        st.info("📁 File upload is temporarily unavailable due to storage configuration issues. You can still use the text context area below.")

context_text = st.text_area(
    "Additional context for Gemini (optional)",
    value=st.session_state.get("manual_context", ""),
    height=120,
)
st.session_state["manual_context"] = context_text

combined_context_parts: List[str] = []
if context_text.strip():
    combined_context_parts.append(context_text.strip())
if st.session_state.get("latest_upload_context"):
    combined_context_parts.append(st.session_state["latest_upload_context"])
combined_context = "\n\n".join(combined_context_parts) if combined_context_parts else None

history_records = chain.get_recent_history(user.get("localId"))
display_messages: List[Dict[str, str]] = []
for record in history_records:
    if {"query", "response"}.issubset(record.keys()):
        display_messages.append({"role": "user", "content": record["query"]})
        if record.get("response"):
            display_messages.append({"role": "assistant", "content": record["response"]})
    else:
        display_messages.append(record)

chat_placeholder = st.container()
with chat_placeholder:
    if display_messages:
        chat_components.render_chat_history(display_messages)
    else:
        chat_components.render_empty_state()

prompt = st.chat_input("Ask CareerSaathi anything about your career...")
if prompt:
    chat_components.render_user_message(prompt)
    metadata = {"has_context": bool(combined_context)}
    response = chain.invoke(
        user.get("localId"),
        prompt,
        context=combined_context,
        metadata=metadata,
    )
    chat_components.render_response_message(response)
