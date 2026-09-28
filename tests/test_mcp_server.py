from __future__ import annotations

from types import ModuleType

from knowledge_fabric.mcp.server import create_mcp_server


class _FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tool_names: list[str] = []
        self.tools: dict[str, object] = {}

    def tool(self, name: str):
        def _decorator(func):
            self.tool_names.append(name)
            self.tools[name] = func
            return func

        return _decorator

    def run(self) -> None:
        return


class _FakeTools:
    @staticmethod
    def health_check(**kwargs):
        return {"status": "ok", "kwargs": kwargs}

    @staticmethod
    def list_sources(**kwargs):
        return {"sources": [], "kwargs": kwargs}

    @staticmethod
    def retrieve_evidence(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    @staticmethod
    def get_document(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    @staticmethod
    def explain_retrieval(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    @staticmethod
    def get_index_status(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    @staticmethod
    def get_evidence(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    @staticmethod
    def check_consistency(**kwargs):
        return {"ok": True, "kwargs": kwargs}


def test_create_mcp_server_registers_expected_tools(monkeypatch) -> None:
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    monkeypatch.setitem(__import__("sys").modules, "mcp.server.fastmcp", fastmcp_module)

    server = create_mcp_server(_FakeTools())

    assert server.name == "knowledge-fabric"
    assert sorted(server.tool_names) == [
        "check_consistency",
        "explain_retrieval",
        "get_document",
        "get_evidence",
        "get_index_status",
        "health_check",
        "list_sources",
        "retrieve_evidence",
    ]


def test_mcp_server_validates_tenant_boundary(monkeypatch) -> None:
    import pytest

    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    monkeypatch.setitem(__import__("sys").modules, "mcp.server.fastmcp", fastmcp_module)

    server = create_mcp_server(_FakeTools())
    tools = server.tools

    # 1. Valid tenant passes cleanly
    res = tools["list_sources"](tenant_id="tenant-acme-123")
    assert res["kwargs"]["tenant_id"] == "tenant-acme-123"

    res = tools["get_document"](document_id=1, tenant_id="tenant-corp.org")
    assert res["kwargs"]["tenant_id"] == "tenant-corp.org"

    # 2. Path traversal attempts are blocked with ValueError
    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        tools["list_sources"](tenant_id="../../etc/passwd")

    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        tools["get_document"](document_id=1, tenant_id="../malicious")

    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        tools["retrieve_evidence"](query_text="test", tenant_id="bad tenant with spaces")


def test_mcp_server_forwards_mode_parameter(monkeypatch) -> None:
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    monkeypatch.setitem(__import__("sys").modules, "mcp.server.fastmcp", fastmcp_module)
    monkeypatch.setenv("KF_DEFAULT_TENANT", "default")

    server = create_mcp_server(_FakeTools())
    tools = server.tools

    res_default = tools["retrieve_evidence"](query_text="test")
    assert res_default["kwargs"]["mode"] == "hybrid"

    res_lexical = tools["retrieve_evidence"](query_text="test", mode="lexical")
    assert res_lexical["kwargs"]["mode"] == "lexical"

    res_explain = tools["explain_retrieval"](query_text="test", mode="vector")
    assert res_explain["kwargs"]["mode"] == "vector"


def test_mcp_server_fails_closed_when_tenant_missing(monkeypatch) -> None:
    import pytest

    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    monkeypatch.setitem(__import__("sys").modules, "mcp.server.fastmcp", fastmcp_module)
    monkeypatch.delenv("KF_DEFAULT_TENANT", raising=False)

    server = create_mcp_server(_FakeTools())
    tools = server.tools

    with pytest.raises(ValueError, match="tenant_id is required"):
        tools["retrieve_evidence"](query_text="test")

    with pytest.raises(ValueError, match="tenant_id is required"):
        tools["list_sources"]()

    with pytest.raises(ValueError, match="tenant_id is required"):
        tools["get_document"](document_id=1)

    with pytest.raises(ValueError, match="tenant_id is required"):
        tools["get_index_status"]()

    # health_check does NOT require tenant
    health = tools["health_check"]()
    assert health["status"] == "ok"


