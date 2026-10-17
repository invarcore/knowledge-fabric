# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Database connection helpers."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

from knowledge_fabric.config import DatabaseSettings

ConnectionFactory = Callable[[], Any]

# Prohibited superuser usernames that bypass PostgreSQL Row-Level Security (RLS)
PROHIBITED_SUPERUSER_ROLES = frozenset({"postgres", "root", "superuser", "admin"})


def create_postgres_connection_factory(
    settings: DatabaseSettings,
    *,
    enforce_non_superuser: bool | None = None,
) -> ConnectionFactory:
    """Create a lazy psycopg connection factory from database settings.

    Restricts PostgreSQL connection roles to least-privilege tenant roles rather
    than superuser to prevent Row-Level Security (RLS) bypass (KF-01).
    """

    def _connect() -> Any:
        import psycopg

        # Determine non-superuser enforcement
        should_enforce = enforce_non_superuser
        if should_enforce is None:
            should_enforce = os.environ.get("KNOWLEDGE_ENFORCE_NON_SUPERUSER", "true").lower() not in (
                "false",
                "0",
                "off",
                "no",
            )

        if should_enforce and settings.user.lower() in PROHIBITED_SUPERUSER_ROLES:
            raise PermissionError(
                f"PostgreSQL connection role '{settings.user}' is a prohibited superuser role. "
                "Knowledge Fabric requires a dedicated least-privilege tenant role (e.g., 'kf_app') "
                "to ensure Row-Level Security (RLS) policies are enforced (KF-01). "
                "Set KNOWLEDGE_ENFORCE_NON_SUPERUSER=false to override for migrations/admin tooling."
            )

        connect_timeout = int(os.environ.get("KNOWLEDGE_DB_CONNECT_TIMEOUT", "2"))
        conn = psycopg.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.name,
            user=settings.user,
            password=settings.password,
            connect_timeout=connect_timeout,
            autocommit=False,
        )

        if should_enforce:
            try:
                with conn.cursor() as cur:
                    cur.execute("SHOW is_superuser")
                    row = cur.fetchone()
                    if row and str(row[0]).lower() in ("on", "true", "1", "yes"):
                        conn.close()
                        raise PermissionError(
                            f"PostgreSQL connection role '{settings.user}' has superuser privileges (is_superuser=on). "
                            "Knowledge Fabric requires a dedicated least-privilege tenant role (e.g., 'kf_app') "
                            "to ensure Row-Level Security (RLS) policies are enforced (KF-01). "
                            "Set KNOWLEDGE_ENFORCE_NON_SUPERUSER=false to override for migrations/admin tooling."
                        )
            except PermissionError:
                raise
            except Exception:
                # If SHOW is_superuser is unsupported or fails for an unexpected non-permission reason, do not block
                pass

        return conn

    return _connect

