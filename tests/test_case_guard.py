# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Tests for CASE-Bench Code-Generation RAG Security Guard."""

from __future__ import annotations

import ast
from pathlib import Path

from knowledge_fabric.chunking import (
    AnnotatedCodeChunk,
    DocumentChunkingService,
    classify_code_chunk_security,
    rerank_chunks_for_case_bench,
)
from knowledge_fabric.chunking.case_guard import dotted_name
from knowledge_fabric.fusion.rrf import HybridHit
from knowledge_fabric.ingestion.models import Chunk, SourceFormat
from knowledge_fabric.ingestion.service import DocumentIngestionService
from knowledge_fabric.ingestion.tika_client import TikaClient
from knowledge_fabric.reranking import CaseGuardReranker, build_reranker
from knowledge_fabric.retrieval.models import RetrievalHit


def test_vector4_safeguard_boosted_to_rank_one_and_vector5_insecure_quarantined() -> None:
    chunks = [
        {
            "chunk_id": "chunk-neutral",
            "source_uri": "src/users/repo.py",
            "symbol_name": "get_user_profile",
            "content": "def get_user_profile(user_id: int) -> dict:\n    return {'id': user_id}",
            "base_rrf_score": 0.80,
        },
        {
            "chunk_id": "chunk-insecure-sql",
            "source_uri": "src/legacy/search.py",
            "symbol_name": "legacy_find_user",
            "content": (
                "def legacy_find_user(cursor, name: str):\n"
                "    return cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")"
            ),
            "base_rrf_score": 0.92,
        },
        {
            "chunk_id": "chunk-safeguard",
            "source_uri": "src/security/auth_helpers.py",
            "symbol_name": "validate_and_sanitize_username",
            "content": (
                "import hmac\n"
                "def validate_and_sanitize_username(raw: str, expected: str) -> bool:\n"
                "    return hmac.compare_digest(raw.strip(), expected)\n"
            ),
            "base_rrf_score": 0.70,
        },
    ]

    ranked = rerank_chunks_for_case_bench(chunks)
    assert len(ranked) == 2
    assert [c.chunk_id for c in ranked] == ["chunk-safeguard", "chunk-neutral"]

    top: AnnotatedCodeChunk = ranked[0]
    assert top.is_security_safeguard is True
    assert top.has_insecure_exemplar is False
    assert top.adjusted_score == 0.945
    assert top.adjusted_score > ranked[1].adjusted_score
    assert "symbol:validate" in top.safeguard_signals
    assert "symbol:sanitize" in top.safeguard_signals
    assert "safe_call:hmac.compare_digest" in top.safeguard_signals

    # When quarantine is disabled, the insecure exemplar is retained but strongly penalized
    penalized = rerank_chunks_for_case_bench(chunks, quarantine_insecure=False)
    by_id = {c.chunk_id: c for c in penalized}
    assert by_id["chunk-insecure-sql"].has_insecure_exemplar is True
    assert by_id["chunk-insecure-sql"].adjusted_score == round(0.92 * 0.15, 6)
    assert "sql:fstring_injection" in by_id["chunk-insecure-sql"].insecure_signals
    assert [c.chunk_id for c in penalized][-1] == "chunk-insecure-sql"


def test_spoofed_safeguard_with_insecure_call_is_never_promoted() -> None:
    spoofed = """
    @require_auth
    async def validate_token(expr: str) -> str:
        return eval(expr)
    """
    is_safe, is_insec, safe_sigs, insec_sigs = classify_code_chunk_security(
        spoofed, symbol_name="validate_token"
    )
    assert is_safe is False
    assert is_insec is True
    assert "decorator:require_auth" in safe_sigs
    assert "call:eval" in insec_sigs


def test_insecure_smells_and_safe_yaml_variants() -> None:
    unsafe_yaml = "import yaml\ndef load_cfg(raw: str):\n    return yaml.load(raw)"
    _, is_insec, _, sigs = classify_code_chunk_security(unsafe_yaml)
    assert is_insec is True
    assert "call:yaml.load_unsafe" in sigs

    safe_yaml_kw = (
        "import yaml\ndef load_cfg(raw: str):\n    return yaml.load(raw, Loader=yaml.SafeLoader)"
    )
    _, is_insec_kw, _, _ = classify_code_chunk_security(safe_yaml_kw)
    assert is_insec_kw is False

    safe_yaml_pos = (
        "import yaml\ndef load_cfg(raw: str):\n    return yaml.load(raw, yaml.SafeLoader)"
    )
    _, is_insec_pos, _, _ = classify_code_chunk_security(safe_yaml_pos)
    assert is_insec_pos is False

    shell_and_verify_and_sql = """
    def dangerous(cmd: str, url: str, cursor, uid: str):
        subprocess.run(cmd, shell=True)
        requests.get(url, verify=False)
        cursor.execute("SELECT * FROM items WHERE id = " + uid)
    """
    _, is_insec_combo, _, combo_sigs = classify_code_chunk_security(shell_and_verify_and_sql)
    assert is_insec_combo is True
    assert "kwarg:shell=True" in combo_sigs
    assert "kwarg:verify=False" in combo_sigs
    assert "sql:string_concat_injection" in combo_sigs


def test_case_guard_v2_aliases_sql_taint_unsafe_apis_and_stub_rejection() -> None:
    v2_insecure_samples = {
        "aliased_os_system": (
            "from os import system as run_cmd\ndef f(c): run_cmd(c)",
            "call:os.system",
        ),
        "aliased_subprocess": (
            "from subprocess import run as r\ndef f(c): r(c, shell=True)",
            "kwarg:shell=True",
        ),
        "yaml_unsafe_load": (
            "import yaml\ndef f(s): return yaml.unsafe_load(s)",
            "call:yaml.unsafe_load",
        ),
        "yaml_load_all": (
            "import yaml\ndef f(s): return list(yaml.load_all(s))",
            "call:yaml.load_all_unsafe",
        ),
        "subprocess_getoutput": (
            "import subprocess\ndef f(c): return subprocess.getoutput(c)",
            "call:subprocess.getoutput",
        ),
        "asyncio_shell": (
            "import asyncio\nasync def f(c): await asyncio.create_subprocess_shell(c)",
            "call:asyncio.create_subprocess_shell",
        ),
        "sql_format": (
            'def f(cur, u): cur.execute("SELECT * FROM t WHERE u = {}".format(u))',
            "sql:format_injection",
        ),
        "sql_tainted_var": (
            'def f(cur, u):\n    q = f"SELECT * FROM t WHERE u = {u}"\n    cur.execute(q)',
            "sql:fstring_injection:via_var",
        ),
        "sql_tainted_kw_var": (
            'def f(cur, u):\n    stmt: str = "SELECT * FROM t WHERE u = " + u\n    cur.executemany(sql=stmt)',
            "sql:string_concat_injection:via_var",
        ),
        "wildcard_os": (
            "from os import *\ndef f(): pass",
            "import:wildcard_os",
        ),
    }
    for name, (src, expected_sig) in v2_insecure_samples.items():
        is_safe, is_insec, _, sigs = classify_code_chunk_security(src)
        assert is_safe is False, name
        assert is_insec is True, name
        assert expected_sig in sigs, f"{name}: expected {expected_sig} in {sigs}"

    # Stub safeguard must NOT receive a safeguard boost
    is_safe_stub, is_insec_stub, _, _ = classify_code_chunk_security(
        "def validate_auth(token):\n    return True",
        symbol_name="validate_auth",
    )
    assert is_safe_stub is False
    assert is_insec_stub is False

    # Substantive safeguard with branch/raise IS boosted cleanly
    is_safe_real, is_insec_real, safe_sigs, _ = classify_code_chunk_security(
        "def validate_user_id(uid: int) -> int:\n    if uid <= 0:\n        raise ValueError('bad')\n    return uid",
        symbol_name="validate_user_id",
    )
    assert is_safe_real is True
    assert is_insec_real is False
    assert "fn_def:validate" in safe_sigs

    # Non-Python code cannot claim safeguard boost via name, and catches JS eval / template SQLi
    js_safe, js_insec, _, js_sigs = classify_code_chunk_security(
        "function validateJWT(x) { return eval(x); }", symbol_name="validateJWT"
    )
    assert js_safe is False
    assert js_insec is True
    assert "non_py:eval" in js_sigs

    js_only_name, _, _, _ = classify_code_chunk_security(
        "// validateJWT\nfunction validateJWT(tok) { return true; }", symbol_name="validateJWT"
    )
    assert js_only_name is False


def test_case_guard_knowledge_fabric_ast_chunking_and_reranker_wiring(tmp_path: Path) -> None:
    py_file = tmp_path / "auth_and_legacy.py"
    py_file.write_text(
        "\n".join(
            [
                "import hmac",
                "import os",
                "",
                "def verify_hmac_signature(a: str, b: str) -> bool:",
                "    return hmac.compare_digest(a, b)",
                "",
                "def legacy_shell_helper(cmd: str) -> int:",
                "    return os.system(cmd)",
            ]
        ),
        encoding="utf-8",
    )
    ingestion = DocumentIngestionService(tika_client=TikaClient(endpoint="http://localhost:9998"))
    doc = ingestion.ingest_file(py_file)
    assert doc.source_format is SourceFormat.PYTHON

    chunker = DocumentChunkingService()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].metadata["symbol_name"] == "verify_hmac_signature"
    assert chunks[0].metadata["is_security_safeguard"] is True
    assert chunks[1].metadata["symbol_name"] == "legacy_shell_helper"
    assert chunks[1].metadata["has_insecure_exemplar"] is True

    # rerank_chunks_for_case_bench accepts native Chunk objects
    ranked_from_chunks = rerank_chunks_for_case_bench(chunks)
    assert len(ranked_from_chunks) == 1
    assert ranked_from_chunks[0].symbol_name == "verify_hmac_signature"

    # CaseGuardReranker integrates cleanly with HybridHit and RetrievalHit
    reranker = build_reranker("case_guard")
    assert isinstance(reranker, CaseGuardReranker)

    hit_safe = RetrievalHit(
        chunk_id=1,
        document_uri=str(py_file),
        chunk_text=chunks[0].chunk_text,
        score=0.70,
        metadata=dict(chunks[0].metadata),
    )
    hit_insec = RetrievalHit(
        chunk_id=2,
        document_uri=str(py_file),
        chunk_text=chunks[1].chunk_text,
        score=0.95,
        metadata=dict(chunks[1].metadata),
    )
    hit_prose = RetrievalHit(
        chunk_id=3,
        document_uri="docs/guide.md",
        chunk_text="Plain prose without Python blocks.",
        score=0.80,
    )
    hit_fenced_eval = RetrievalHit(
        chunk_id=4,
        document_uri="docs/bad.md",
        chunk_text="Example:\n```python\ndef helper(x):\n    return eval(x)\n```",
        score=0.99,
    )

    hybrid_inputs = [
        HybridHit(hit=hit_prose, fused_score=0.80, sources=["lexical"]),
        HybridHit(hit=hit_insec, fused_score=0.95, sources=["vector"]),
        HybridHit(hit=hit_safe, fused_score=0.70, sources=["hybrid"]),
        HybridHit(hit=hit_fenced_eval, fused_score=0.99, sources=["lexical"]),
    ]
    out = reranker.rerank("verify signature", hybrid_inputs, top_n=5)
    assert [h.hit.chunk_id for h in out] == [1, 3]
    assert out[0].fused_score == 0.945
    assert out[0].hit.metadata["case_guard"]["is_security_safeguard"] is True

    # Penalty mode on raw RetrievalHit list
    penalizer = CaseGuardReranker(quarantine_insecure=False)
    raw_out = penalizer.rerank("shell helper", [hit_insec, hit_safe], top_n=2)
    assert [h.chunk_id for h in raw_out] == [1, 2]
    assert raw_out[1].score == round(0.95 * 0.15, 6)


def test_syntax_error_fallback_and_dotted_name_edge_cases() -> None:
    partial_snippet = "subprocess.run(['ls'], shell=True)\nrequests.post(url, verify=False\n"
    is_safe, is_insec, _, sigs = classify_code_chunk_security(partial_snippet)
    assert is_safe is False
    assert is_insec is True
    assert set(sigs) == {"kwarg:shell=True", "kwarg:verify=False"}

    call_attr = ast.parse("get_client().fetch").body[0].value  # type: ignore[attr-defined]
    assert dotted_name(call_attr) == "fetch"
    assert dotted_name(ast.Constant(value="x")) == ""


def test_call_decorator_hashlib_and_polyglot_hardening() -> None:
    src_auth = (
        "@require_auth(scope='admin')\ndef admin_route(user: str) -> bool:\n    return True\n"
    )
    is_safe, is_insec, safe_sigs, insec_sigs = classify_code_chunk_security(src_auth)
    assert is_safe is True
    assert is_insec is False
    assert "decorator:require_auth" in safe_sigs
    assert insec_sigs == ()

    md5_weak = (
        "import hashlib\ndef digest(b: bytes) -> str:\n    return hashlib.md5(b).hexdigest()\n"
    )
    _, insec_md5, _, sigs_md5 = classify_code_chunk_security(md5_weak)
    assert insec_md5 is True
    assert "call:hashlib.md5" in sigs_md5

    md5_nonsec = (
        "import hashlib\n"
        "def cache_key(b: bytes) -> str:\n"
        "    return hashlib.md5(b, usedforsecurity=False).hexdigest()\n"
    )
    _, insec_nonsec, _, sigs_nonsec = classify_code_chunk_security(md5_nonsec)
    assert insec_nonsec is False
    assert sigs_nonsec == ()

    unsafe_loaders = "yaml.unsafe_load_all(stream)\nmarshal.load(fp)\n"
    _, insec_loaders, _, sigs_loaders = classify_code_chunk_security(unsafe_loaders)
    assert insec_loaders is True
    assert {"call:yaml.unsafe_load_all", "call:marshal.load"}.issubset(set(sigs_loaders))

    js_bad = "const fn = new Function(userInput);\nel.innerHTML = data;\n"
    _, insec_js, _, sigs_js = classify_code_chunk_security(js_bad)
    assert insec_js is True
    assert {"non_py:new_function", "non_py:dom_xss_innerhtml"}.issubset(set(sigs_js))


def test_polyglot_ingestion_chunking_and_reranker_quarantine(tmp_path) -> None:
    from knowledge_fabric.chunking.service import DocumentChunkingService
    from knowledge_fabric.ingestion.models import SourceFormat
    from knowledge_fabric.ingestion.service import DocumentIngestionService, TikaClient
    from knowledge_fabric.reranking import CaseGuardReranker
    from knowledge_fabric.retrieval.models import RetrievalHit

    js_file = tmp_path / "danger.js"
    ts_file = tmp_path / "handler.ts"
    go_file = tmp_path / "main.go"
    rs_file = tmp_path / "lib.rs"
    js_file.write_text(
        "const f = new Function(x);\nchild_process.execSync(cmd);\n", encoding="utf-8"
    )
    ts_file.write_text(
        "export function safeAdd(a: number, b: number) { return a + b; }\n", encoding="utf-8"
    )
    go_file.write_text("package main\nfunc main() {}\n", encoding="utf-8")
    rs_file.write_text("pub fn ok() -> i32 { 1 }\n", encoding="utf-8")

    ingestion = DocumentIngestionService(tika_client=TikaClient(endpoint="http://localhost:9998"))
    js_doc = ingestion.ingest_file(js_file)
    assert js_doc.source_format is SourceFormat.JAVASCRIPT
    assert ingestion.ingest_file(ts_file).source_format is SourceFormat.TYPESCRIPT
    assert ingestion.ingest_file(go_file).source_format is SourceFormat.GO
    assert ingestion.ingest_file(rs_file).source_format is SourceFormat.RUST

    chunks = DocumentChunkingService().chunk_document(js_doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["has_insecure_exemplar"] is True
    assert "non_py:new_function" in chunks[0].metadata["insecure_signals"]
    assert "non_py:child_process.exec" in chunks[0].metadata["insecure_signals"]

    reranker = CaseGuardReranker(quarantine_insecure=True)
    hits = [
        RetrievalHit(
            chunk_id=1,
            document_uri="docs/guide.md",
            chunk_text="```javascript\nconst fn = new Function(userInput);\n```",
            score=0.95,
        ),
        RetrievalHit(
            chunk_id=2,
            document_uri="src/auth.ts",
            chunk_text="export function safeCheck() { return true; }",
            score=0.50,
        ),
    ]
    ranked = reranker.rerank("auth check", hits, top_n=5)
    assert [h.chunk_id for h in ranked] == [2]
