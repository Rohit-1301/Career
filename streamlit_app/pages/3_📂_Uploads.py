"""Manage uploaded files stored in Firebase Storage."""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from firebase import storage
from services.utils import get_logger

from streamlit_app.components import session as session_components

logger = get_logger(__name__)

session_components.initialise_session()
user = session_components.require_authentication()

st.title("📂 Uploaded Files")

uid = user.get("localId")

# Check if Firebase Storage is available
try:
    from firebase.config import get_storage_bucket
    get_storage_bucket()  # Test if bucket is accessible
    storage_available = True
except Exception as e:
    logger.warning(f"Firebase Storage not available: {e}")
    storage_available = False

if not storage_available:
    st.error("🚨 Firebase Storage is not properly configured. File uploads and management are currently unavailable.")
    st.info("🛠️ To fix this, please ensure your Firebase Storage bucket exists and is properly configured.")
    st.stop()

try:
    blobs = storage.list_user_files(uid)
except Exception as exc:  # pragma: no cover - external service dependency
    st.error(f"Could not fetch uploaded files: {exc}")
    blobs = []

if not blobs:
    st.info("No files uploaded yet. Use the dashboard to add supporting documents.")
else:
    for blob in blobs:
        cols = st.columns([4, 2, 2])
        with cols[0]:
            st.write(f"**{blob.name.split('/')[-1]}**")
            st.caption(f"Size: {blob.size / 1024:.1f} KB | Updated: {blob.updated}")
        with cols[1]:
            try:
                download_url = blob.generate_signed_url(expiration=timedelta(minutes=10))
                st.markdown(f"[Download link]({download_url})", unsafe_allow_html=False)
            except Exception as exc:  # pragma: no cover - requires credentials
                st.warning(f"No download link available: {exc}")
        with cols[2]:
            if st.button("Delete", key=f"delete-{blob.name}"):
                try:
                    storage.delete_user_file(uid, blob.name.split("/")[-1])
                    st.success("File deleted")
                    st.rerun()
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to delete file: {exc}")
