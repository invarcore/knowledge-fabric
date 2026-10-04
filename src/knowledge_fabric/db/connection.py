# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Database connection helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from knowledge_fabric.config import DatabaseSettings

ConnectionFactory = Callable[[], Any]


def create_postgres_connection_factory(settings: DatabaseSettings) -> ConnectionFactory:
    """Create a lazy psycopg connection factory from database settings."""

    def _connect() -> Any:
        import os
        import psycopg

        connect_timeout = int(os.environ.get("KNOWLEDGE_DB_CONNECT_TIMEOUT", "2"))
        return psycopg.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.name,
            user=settings.user,
            password=settings.password,
            connect_timeout=connect_timeout,
            autocommit=False,
        )

    return _connect
