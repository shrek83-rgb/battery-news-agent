from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _get_drive_service():
    key_json = os.getenv("GDRIVE_SA_KEY_JSON", "")
    if not key_json:
        raise RuntimeError("Missing env: GDRIVE_SA_KEY_JSON")

    info = json.loads(key_json)
    creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def ensure_date_folder(parent_folder_id: str, date_str: str, supports_all_drives: bool = True) -> str:
    """
    Ensure a folder named date_str exists under parent_folder_id.
    Return folder_id.
    """
    service = _get_drive_service()

    q = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and name='{date_str}' "
        f"and '{parent_folder_id}' in parents "
        f"and trashed=false"
    )
    res = service.files().list(
        q=q,
        fields="files(id, name)",
        pageSize=10,
        supportsAllDrives=supports_all_drives,
        includeItemsFromAllDrives=supports_all_drives,
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]

    folder_metadata = {
        "name": date_str,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(
        body=folder_metadata,
        fields="id",
        supportsAllDrives=supports_all_drives,
    ).execute()
    return folder["id"]


def upload_or_update_file(
    file_path: Path,
    folder_id: str,
    mime_type: str,
    supports_all_drives: bool = True,
) -> str:
    """
    Upload file to Drive folder. If a file with same name exists in folder, update it.
    Returns file_id.
    """
    service = _get_drive_service()
    name = file_path.name

    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(
        q=q,
        fields="files(id, name)",
        pageSize=10,
        supportsAllDrives=supports_all_drives,
        includeItemsFromAllDrives=supports_all_drives,
    ).execute()
    existing = res.get("files", [])

    media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)

    if existing:
        file_id = existing[0]["id"]
        updated = service.files().update(
            fileId=file_id,
            media_body=media,
            fields="id",
            supportsAllDrives=supports_all_drives,
        ).execute()
        return updated["id"]

    meta = {"name": name, "parents": [folder_id]}
    created = service.files().create(
        body=meta,
        media_body=media,
        fields="id",
        supportsAllDrives=supports_all_drives,
    ).execute()
    return created["id"]
