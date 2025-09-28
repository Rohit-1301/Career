"""Firestore database helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from firebase_admin import firestore

from services.utils import get_logger

from .config import get_firestore_client

logger = get_logger(__name__)

USER_COLLECTION = "users"
HISTORY_SUBCOLLECTION = "history"


def _user_doc(uid: str) -> firestore.DocumentReference:
    return get_firestore_client().collection(USER_COLLECTION).document(uid)


def upsert_user_profile(uid: str, profile: Dict[str, Any]) -> None:
    """Create or update a user profile document."""
    doc = _user_doc(uid)
    profile.setdefault("updatedAt", firestore.SERVER_TIMESTAMP)
    if "createdAt" not in profile:
        profile["createdAt"] = firestore.SERVER_TIMESTAMP
    logger.info("Upserting profile for user %s", uid)
    doc.set(profile, merge=True)


def get_user_profile(uid: str) -> Optional[Dict[str, Any]]:
    doc = _user_doc(uid).get()
    return doc.to_dict() if doc.exists else None


def save_history_entry(
    uid: str,
    query: str,
    response: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a single query/response pair for a user."""
    metadata = metadata or {}
    history_ref = _user_doc(uid).collection(HISTORY_SUBCOLLECTION)
    data = {
        "query": query,
        "response": response,
        "metadata": metadata,
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    doc_ref = history_ref.add(data)[1]  # add returns (update_time, document_reference)
    logger.info("Saved history entry %s for user %s", doc_ref.id, uid)
    return doc_ref.id


def list_history(uid: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve the most recent history entries for a user."""
    history_ref = (
        _user_doc(uid)
        .collection(HISTORY_SUBCOLLECTION)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    entries = [doc.to_dict() | {"id": doc.id} for doc in history_ref.stream()]
    return entries


def record_file_reference(uid: str, file_path: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Store a pointer to a user-uploaded file in Firestore."""
    files_ref = _user_doc(uid).collection("files")
    data = {
        "path": file_path,
        "metadata": metadata or {},
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    files_ref.add(data)
    logger.info("Recorded file reference for user %s at %s", uid, file_path)
