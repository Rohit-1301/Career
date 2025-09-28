"""Firebase Storage helpers."""

from __future__ import annotations

import io
from datetime import timedelta
from typing import BinaryIO, List, Optional

from google.cloud.storage import Bucket
from google.cloud import storage

from services.utils import get_logger

from .config import get_storage_bucket

logger = get_logger(__name__)


def _get_bucket() -> Bucket:
    return get_storage_bucket()


def _ensure_stream(file_obj: BinaryIO | bytes) -> BinaryIO:
    if isinstance(file_obj, (bytes, bytearray)):
        return io.BytesIO(file_obj)
    return file_obj


def upload_user_file(
    uid: str,
    file_obj: BinaryIO | bytes,
    filename: str,
    *,
    content_type: Optional[str] = None,
    make_public: bool = False,
) -> dict:
    """Upload a file to Firebase Storage under the user's namespace."""
    try:
        bucket = get_storage_bucket()
    except Exception as e:
        logger.error(f"Failed to get storage bucket: {e}")
        raise RuntimeError(
            f"Firebase Storage is not properly configured. "
            f"Please check your bucket configuration. Error: {e}"
        ) from e
    
    blob_path = f"users/{uid}/{filename}"
    blob = bucket.blob(blob_path)
    stream = _ensure_stream(file_obj)
    stream.seek(0)
    
    try:
        blob.upload_from_file(stream, content_type=content_type, rewind=True)
        logger.info("Uploaded file %s for user %s", blob_path, uid)
    except Exception as e:
        logger.error(f"Failed to upload file {filename} for user {uid}: {e}")
        raise RuntimeError(f"File upload failed: {e}") from e

    download_url: Optional[str] = None
    if make_public:
        blob.make_public()
        download_url = blob.public_url
    else:
        try:
            download_url = blob.generate_signed_url(expiration=timedelta(hours=1))
        except Exception as exc:  # pragma: no cover - requires credentials
            logger.warning("Failed to generate signed URL: %s", exc)

    return {
        "path": blob_path,
        "gs_uri": f"gs://{bucket.name}/{blob_path}",
        "download_url": download_url,
    }


def delete_user_file(uid: str, filename: str) -> None:
    bucket = get_storage_bucket()
    blob_path = f"users/{uid}/{filename}"
    blob = bucket.blob(blob_path)
    blob.delete()
    logger.info("Deleted file %s for user %s", blob_path, uid)


def list_user_files(uid: str) -> list[storage.blob.Blob]:
    bucket = get_storage_bucket()
    prefix = f"users/{uid}/"
    logger.info("Listing files for user %s", uid)
    return list(bucket.list_blobs(prefix=prefix))
