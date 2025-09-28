"""Entry point for the CareerSaathi Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from firebase import auth as firebase_auth
from services.utils import get_logger

from streamlit_app.components import auth as auth_components
from streamlit_app.components import session as session_components
from streamlit_app.components.sidebar import render_authenticated_sidebar

logger = get_logger(__name__)

st.set_page_config(page_title="CareerSaathi AI Coach", page_icon="🧭", layout="wide")

session_components.initialise_session()


def render_authenticated_home() -> None:
    user = session_components.get_current_user()
    if not user:
        return

    sign_out = render_authenticated_sidebar(user)
    if sign_out:
        session_components.clear_session()
        st.rerun()

    st.title("Welcome to CareerSaathi 👋")
    st.caption("Your AI-powered career coach powered by Gemini and Firebase.")

    st.write(
        """
        Use the sidebar to navigate between pages:

        - **🏠 Dashboard** — Ask the AI coach questions, upload supporting documents,
          and receive personalised guidance powered by LangChain + Gemini.
        - **📜 History** — Revisit your previous conversations and insights.
        - **📂 Uploads** — Manage the files you've shared with CareerSaathi.

        You can sign out at any time using the sidebar.
        """
    )


if session_components.get_current_user():
    render_authenticated_home()
else:
    st.title("CareerSaathi Login")
    st.write("Sign in or create an account to unlock your personalised career coach.")
    login_tab, signup_tab = st.tabs(["Login", "Sign up"])

    with login_tab:
        login_result = auth_components.render_login_form()
        if login_result.success and login_result.payload:
            session_components.set_current_user(login_result.payload)
            st.success("Logged in successfully! Redirecting...")
            st.rerun()

    with signup_tab:
        signup_result = auth_components.render_signup_form()
        if signup_result.success and signup_result.payload:
            session_components.set_current_user(signup_result.payload)
            st.success("Account created and signed in! Redirecting...")
            st.rerun()

    st.caption("By continuing you agree to our terms of service and privacy policy.")
