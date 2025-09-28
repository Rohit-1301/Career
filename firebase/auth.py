"""Authentication helpers built on Firebase Auth."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from firebase_admin import auth as admin_auth

from services.utils import get_logger

from .config import get_auth_client, get_firestore_client, get_admin_app
from .db import upsert_user_profile

logger = get_logger(__name__)


class AuthError(Exception):
    """Raised when Firebase authentication fails."""


def _handle_error(exc: Exception) -> None:
    logger.error("Firebase auth error: %s", exc)
    raise AuthError(str(exc)) from exc


def sign_up(
    email: str,
    password: str,
    *,
    display_name: Optional[str] = None,
    send_verification_email: bool = True,
) -> Dict[str, Any]:
    """Create a new Firebase user using email/password credentials."""
    auth_client = get_auth_client()
    try:
        user = auth_client.create_user_with_email_and_password(email, password)
    except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
        _handle_error(exc)
    else:
        uid = user.get("localId")
        if not uid:
            raise AuthError("Firebase did not return a user id (localId)")

        get_admin_app()  # Ensure admin app is initialized before use
        if display_name:
            admin_auth.update_user(uid, display_name=display_name)
        upsert_user_profile(
            uid,
            {
                "email": email,
                "displayName": display_name,
                "emailVerified": bool(user.get("emailVerified")),
            },
        )

        if send_verification_email:
            try:
                auth_client.send_email_verification(user["idToken"])
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to send verification email: %s", exc)

        return user


def sign_in(email: str, password: str) -> Dict[str, Any]:
    """Authenticate a user and return the Firebase session payload."""
    auth_client = get_auth_client()
    try:
        user = auth_client.sign_in_with_email_and_password(email, password)
    except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
        _handle_error(exc)
    return user


def refresh_id_token(refresh_token: str) -> Dict[str, Any]:
    auth_client = get_auth_client()
    try:
        return auth_client.refresh(refresh_token)
    except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
        _handle_error(exc)
    return {}


def verify_id_token(id_token: str) -> Dict[str, Any]:
    try:
        return admin_auth.verify_id_token(id_token)
    except Exception as exc:  # pragma: no cover - external dependency
        _handle_error(exc)
    return {}


def send_password_reset(email: str) -> None:
    auth_client = get_auth_client()
    try:
        auth_client.send_password_reset_email(email)
    except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
        _handle_error(exc)
