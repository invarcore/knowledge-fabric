# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

import pytest

from knowledge_fabric.db.connection_pool import (
    RLSSafeConnectionPool,
    SuperuserViolationError,
)


class FakeCursor:
    def __init__(self, conn, fail_execute: bool = False):
        self.conn = conn
        self.fail_execute = fail_execute

    def execute(self, sql: str) -> None:
        if self.fail_execute and "set_config" in sql:
            raise RuntimeError("Database connection reset")
        self.conn.executed_sql.append(sql)
        if "set_config('app.tenant_id', ''" in sql:
            self.conn.active_tenant = ""

    def fetchone(self):
        return (self.conn.superuser_flag,)

    def close(self) -> None:
        pass


class FakeConn:
    def __init__(self, superuser_flag: str = "off", fail_on_scrub: bool = False) -> None:
        self.closed = False
        self.superuser_flag = superuser_flag
        self.active_tenant = "tenant-secret-a"
        self.executed_sql = []
        self.fail_on_scrub = fail_on_scrub

    def cursor(self):
        return FakeCursor(self, fail_execute=self.fail_on_scrub)

    def rollback(self) -> None:
        self.executed_sql.append("ROLLBACK")

    def commit(self) -> None:
        self.executed_sql.append("COMMIT")

    def close(self) -> None:
        self.closed = True


def test_rls_safe_connection_pool_reuses_socket_and_scrubs_tenant() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=4)
    for _ in range(3):
        c = pool.acquire()
        c.active_tenant = "tenant-a"
        pool.release(c)

    assert pool.physical_connect_count == 1
    assert pool.superuser_check_count == 1
    c_final = pool.acquire()
    assert c_final.active_tenant == ""

    # Superuser check failure
    bad_pool = RLSSafeConnectionPool(raw_connect_fn=lambda: FakeConn("on"))
    with pytest.raises(SuperuserViolationError, match="PostgreSQL superuser"):
        bad_pool.acquire()


def test_pool_connection_scrub_failure_and_closed_socket() -> None:
    failing_conn = FakeConn("off", fail_on_scrub=True)
    pool = RLSSafeConnectionPool(raw_connect_fn=lambda: failing_conn, max_idle=2)
    c = pool.acquire()
    pool.release(c)
    # Since scrubbing failed, failing_conn must have been closed and not added to pool
    assert failing_conn.closed is True
    assert len(pool._idle_pool) == 0

    # Release None or closed connection does nothing
    pool.release(None)
    closed_conn = FakeConn("off")
    closed_conn.closed = True
    pool.release(closed_conn)


def test_pool_capacity_overflow_closes_excess() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=2)
    c1 = pool.acquire()
    c2 = pool.acquire()
    c3 = pool.acquire()
    pool.release(c1)
    pool.release(c2)
    pool.release(c3)
    assert len(pool._idle_pool) == 2
    assert c3.closed is True


def test_pool_acquire_skips_closed_idle_connections() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=4)
    c = pool.acquire()
    pool.release(c)
    assert len(pool._idle_pool) == 1
    # Manually simulate socket closed by server
    c.closed = True
    c_new = pool.acquire()
    assert c_new is not c
    assert pool.physical_connect_count == 2


def test_superuser_check_exception_handling() -> None:
    class CrashingCursorConn(FakeConn):
        def cursor(self):
            raise ConnectionResetError("network timeout")

    crashing_pool = RLSSafeConnectionPool(raw_connect_fn=CrashingCursorConn)
    with pytest.raises(RuntimeError, match="Failed superuser verification"):
        crashing_pool.acquire()


def test_scrub_failure_with_failing_close() -> None:
    class FailingCloseConn(FakeConn):
        def close(self):
            raise RuntimeError("close failed")

    pool = RLSSafeConnectionPool(raw_connect_fn=lambda: FailingCloseConn("off", fail_on_scrub=True))
    c = pool.acquire()
    pool.release(c)
    assert len(pool._idle_pool) == 0


def test_max_active_exhaustion_and_timeout() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=2, max_active=2)
    c1 = pool.acquire()
    c2 = pool.acquire()
    assert pool.active_checkouts == 2

    # Third acquire times out
    with pytest.raises(TimeoutError, match="Connection pool exhausted: 2 active"):
        pool.acquire(timeout_s=0.01)

    # Releasing c1 frees a slot
    pool.release(c1)
    assert pool.active_checkouts == 1
    c3 = pool.acquire(timeout_s=0.1)
    assert pool.active_checkouts == 2
    pool.release(c2)
    pool.release(c3)
    assert pool.active_checkouts == 0


def test_connection_context_manager() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=2, max_active=2)
    with pool.connection() as conn:
        assert pool.active_checkouts == 1
        assert conn.active_tenant == "tenant-secret-a"
    assert pool.active_checkouts == 0

    # Exception inside context manager still releases
    with pytest.raises(ValueError, match="simulated failure"):
        with pool.connection() as conn:
            raise ValueError("simulated failure")
    assert pool.active_checkouts == 0


def test_untracked_and_none_release() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=2, max_active=2)
    pool.release(None)
    untracked = FakeConn("off")
    pool.release(untracked)
    assert pool.active_checkouts == 0


def test_release_closed_checked_out_connection() -> None:
    pool = RLSSafeConnectionPool(raw_connect_fn=FakeConn, max_idle=2, max_active=2)
    conn = pool.acquire()
    assert pool.active_checkouts == 1
    # Caller or network closes connection while checked out
    conn.closed = True
    pool.release(conn)
    assert pool.active_checkouts == 0
    assert len(pool._idle_pool) == 0


def test_acquire_creation_failure_releases_semaphore() -> None:
    pool = RLSSafeConnectionPool(
        raw_connect_fn=lambda: FakeConn("on"),  # Fails superuser check
        max_active=1,
    )
    with pytest.raises(SuperuserViolationError):
        pool.acquire()
    # Ensure semaphore was released and not permanently locked
    assert pool.active_checkouts == 0
