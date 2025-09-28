"""Centralised Firebase configuration helpers."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

import firebase_admin
import pyrebase
from firebase_admin import credentials, firestore, storage as admin_storage

from services.utils import get_env_variable, get_logger

logger = get_logger(__name__)


def _build_pyrebase_config() -> Dict[str, str]:
    keys = [
        "apiKey",
        "authDomain",
        "databaseURL",
        "projectId",
        "storageBucket",
        "messagingSenderId",
        "appId",
        "measurementId",
    ]
    env_mapping = {
        "apiKey": get_env_variable("FIREBASE_API_KEY", required=True),
        "authDomain": get_env_variable("FIREBASE_AUTH_DOMAIN", required=True),
        "databaseURL": get_env_variable("FIREBASE_DATABASE_URL"),
        "projectId": get_env_variable("FIREBASE_PROJECT_ID", required=True),
        "storageBucket": get_env_variable("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": get_env_variable("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": get_env_variable("FIREBASE_APP_ID", required=True),
        "measurementId": get_env_variable("FIREBASE_MEASUREMENT_ID"),
    }

    missing = [key for key in keys if env_mapping.get(key) is None and key in {"apiKey", "authDomain", "projectId", "appId"}]
    if missing:
        raise RuntimeError(f"Missing required Firebase configuration keys: {', '.join(missing)}")

    return env_mapping


@lru_cache(maxsize=1)
def get_pyrebase_app() -> pyrebase.pyrebase.Pyrebase:
    """Return the singleton Pyrebase app instance."""
    config = _build_pyrebase_config()
    logger.info("Initializing Pyrebase client")
    return pyrebase.initialize_app(config)


def _get_service_account_credentials() -> credentials.Base:
    cred_path = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS", required=True)
    # Normalize the path to handle Windows backslashes properly
    cred_path = os.path.normpath(cred_path)
    with open(cred_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return credentials.Certificate(data)


@lru_cache(maxsize=1)
def get_admin_app() -> firebase_admin.App:
    """Return the singleton Firebase Admin SDK app."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        cred = _get_service_account_credentials()
        options: Dict[str, Any] = {}
        bucket = get_env_variable("FIREBASE_STORAGE_BUCKET")
        if bucket:
            options["storageBucket"] = bucket
        logger.info("Initializing Firebase Admin app")
        return firebase_admin.initialize_app(cred, options or None)


def get_auth_client():
    """Return the Pyrebase auth client."""
    return get_pyrebase_app().auth()


@lru_cache(maxsize=1)
def get_firestore_client() -> firestore.Client:
    """Return the singleton Firestore client instance."""
    get_admin_app()  # Ensures the admin app is initialized
    return firestore.client()


def get_storage_bucket() -> admin_storage.bucket.Bucket:
    """Return a Firebase Storage bucket client with fallback bucket name resolution."""
    app = get_admin_app()
    project_id = get_env_variable("FIREBASE_PROJECT_ID", required=True)
    bucket_name = get_env_variable("FIREBASE_STORAGE_BUCKET")
    
    # List of possible bucket names to try
    possible_buckets = []
    if bucket_name:
        possible_buckets.append(bucket_name)
    
    # Add common Firebase bucket naming patterns
    possible_buckets.extend([
        f"{project_id}.appspot.com",
        f"{project_id}.firebasestorage.app",
        project_id
    ])
    
    # Remove duplicates while preserving order
    unique_buckets = list(dict.fromkeys(possible_buckets))
    
    last_error = None
    for bucket_attempt in unique_buckets:
        try:
            logger.info(f"Attempting to connect to bucket: {bucket_attempt}")
            bucket = admin_storage.bucket(bucket_attempt, app=app)
            exists = bucket.exists()
            if not exists:
                logger.warning(
                    "Bucket '%s' does not exist or is not accessible with current credentials.",
                    bucket_attempt,
                )
                continue
            logger.info(f"Successfully connected to bucket: {bucket_attempt}")
            return bucket
        except Exception as e:
            logger.warning(f"Failed to connect to bucket '{bucket_attempt}': {e}")
            last_error = e
            continue
    
    # If all attempts failed, raise the last error with helpful message
    raise RuntimeError(
        f"Unable to connect to any Firebase Storage bucket. Tried: {unique_buckets}. "
        f"Last error: {last_error}. Please ensure the bucket exists and you have proper permissions."
    )
