# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Apache Tika client for binary extraction."""

from __future__ import annotations

import mimetypes
from pathlib import Path

import requests


class TikaClient:
    """Minimal client for text extraction via Apache Tika."""

    def __init__(self, endpoint: str, timeout_seconds: int = 30) -> None:
        clean = endpoint.rstrip("/")
        if clean.endswith("/tika"):
            self._endpoint = clean
        else:
            self._endpoint = f"{clean}/tika"
        self._timeout_seconds = timeout_seconds

    def extract_text(self, file_path: Path) -> str:
        """Extract plain text from binary documents using Tika."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_type, _ = mimetypes.guess_type(file_path.name)
        headers = {
            "Accept": "text/plain",
            "Content-Type": content_type or "application/octet-stream",
        }
        with file_path.open("rb") as source:
            response = requests.put(
                self._endpoint,
                data=source,
                headers=headers,
                timeout=self._timeout_seconds,
            )

        response.raise_for_status()
        return response.text
