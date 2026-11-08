# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""RLS-Safe Connection Pool with Single-Pass Superuser Cache and Tenant State Scrubbing (P1-3)."""

from __future__ import annotations

import collections
import operator
import threading
from typing import Any, Callable, Deque


class SuperuserViolationError(PermissionError):
    """Raised when a pooled PostgreSQL connection runs with superuser privileges."""


class RLSSafeConnectionPool:
    """Knowledge Fabric P1-3 Remediation:

    - Eliminates the 3x TCP handshake + 3x 'SHOW is_superuser' latency penalty per
      retrieve_evidence call (lexical + vector + audit legs).
    - Verifies 'SHOW is_superuser' ONCE when a physical socket is opened.
    - Mandatory Security Invariant (KF-01 Pool Guard): Upon every connection return,
      executes `conn.rollback()` and `SELECT set_config('app.tenant_id', '', false)`
      so tenant RLS context can NEVER bleed across pooled checkouts.
    """

    def __init__(
        self,
        raw_connect_fn: Callable[[], Any],
        max_idle: int = 8,
        enforce_non_superuser: bool = True,
    ) -> None:
        self._raw_connect_fn = raw_connect_fn
        self._max_idle = max_idle
        self._enforce_non_superuser = enforce_non_superuser
        self._idle_pool: Deque[Any] = collections.deque()
        self._lock = threading.Lock()
        self.physical_connect_count: int = 0
        self.superuser_check_count: int = 0

    def _create_verified_connection(self) -> Any:
        conn = self._raw_connect_fn()
        self.physical_connect_count += 1
        if self._enforce_non_superuser:
            try:
                cur = conn.cursor()
                cur.execute("SHOW is_superuser")
                row = cur.fetchone()
                cur.close()
                self.superuser_check_count += 1
                val = str(row[0]).strip().lower() if row else "on"
                if val in ("on", "true", "1", "yes"):
                    conn.close()
                    raise SuperuserViolationError(
                        "KF-01 violation: connection is running as PostgreSQL superuser"
                    )
            except SuperuserViolationError:
                raise
            except Exception as exc:
                conn.close()
                raise RuntimeError(f"Failed superuser verification on connection: {exc}") from exc
        return conn

    def _scrub_tenant_state(self, conn: Any) -> bool:
        try:
            conn.rollback()
            cur = conn.cursor()
            cur.execute("SELECT set_config('app.tenant_id', '', false)")
            cur.close()
            conn.commit()
            return True
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return False

    def acquire(self) -> Any:
        with self._lock:
            while self._idle_pool:
                conn = self._idle_pool.popleft()
                if not getattr(conn, "closed", False):
                    return conn
        return self._create_verified_connection()

    def release(self, conn: Any) -> None:
        if conn is None or getattr(conn, "closed", False):
            return
        if not self._scrub_tenant_state(conn):
            return
        with self._lock:
            if operator.lt(len(self._idle_pool), self._max_idle):
                self._idle_pool.append(conn)
            else:
                conn.close()
