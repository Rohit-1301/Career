"""Reusable authentication widgets for Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st

from firebase import auth as firebase_auth
from firebase.auth import AuthError
from services.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AuthResult:
    success: bool
    payload: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


def render_login_form() -> AuthResult:
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Log in to CareerSaathi")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        try:
            user = firebase_auth.sign_in(email, password)
            logger.info("User %s logged in", email)
            return AuthResult(success=True, payload=user)
        except AuthError as exc:
            st.error(f"Login failed: {exc}")
            return AuthResult(success=False, message=str(exc))
    return AuthResult(success=False)


def render_signup_form() -> AuthResult:
    with st.form("signup_form", clear_on_submit=False):
        st.markdown("### Create a new account")
        display_name = st.text_input("Full name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm password", type="password")
        agree = st.checkbox("I accept the Terms and Privacy Policy")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        if not agree:
            st.warning("Please accept the terms to continue.")
            return AuthResult(success=False, message="terms_not_accepted")
        if password != confirm_password:
            st.error("Passwords do not match.")
            return AuthResult(success=False, message="password_mismatch")
        try:
            signup_payload = firebase_auth.sign_up(email, password, display_name=display_name)
            # Immediately sign in to fetch a refresh token/session payload.
            user_session = firebase_auth.sign_in(email, password)
            st.success("Account created! Please verify your email inbox.")
            logger.info("Created user %s", email)
            return AuthResult(success=True, payload=user_session)
        except AuthError as exc:
            st.error(f"Sign-up failed: {exc}")
            return AuthResult(success=False, message=str(exc))
    return AuthResult(success=False)
