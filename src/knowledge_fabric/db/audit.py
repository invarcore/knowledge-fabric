# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Audit and retrieval run logging."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ConnectionFactory = Callable[[], Any]


class AuditLogger:
    """Writes retrieval telemetry into `retrieval_runs` and `audit_events`."""

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._connection_factory = connection_factory

    def log_retrieval(
        self,
        *,
        query_text: str,
        top_k: int,
        retrieval_mode: str,
        result_count: int,
        latency_ms: int,
        trace_id: str | None = None,
        tenant_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        effective_tenant = tenant_id or "default"
        details_payload = details or {}
        connection = self._connection_factory()
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", [effective_tenant])
            cursor.execute(
                """
                INSERT INTO retrieval_runs (tenant_id, query_text, top_k, retrieval_mode, filters, latency_ms, result_count)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                [effective_tenant, query_text, top_k, retrieval_mode, "{}", latency_ms, result_count],
            )
            cursor.execute(
                """
                INSERT INTO audit_events (tenant_id, event_type, actor, trace_id, event_payload)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                [effective_tenant, "retrieval.executed", "system", trace_id, _json_dump(details_payload)],
            )
            if hasattr(connection, "commit"):
                connection.commit()


def _json_dump(payload: dict[str, object]) -> str:
    # Late import keeps baseline runtime dependency minimal unless logging is used.
    import json

    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
