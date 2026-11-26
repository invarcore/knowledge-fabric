from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pytest

from knowledge_fabric.chunking.case_guard import (
    classify_code_chunk_security,
    rerank_chunks_for_case_bench,
)


@pytest.mark.parametrize(
    "fragment",
    [
        "else:\n    os.system(user_cmd)\n",
        "elif mode == 'fast':\n    pickle.loads(raw_blob)\n",
        "except Exception:\n    builtins.eval(payload)\n",
        "else:\n    getattr(os, 'system')(cmd)\n",
    ],
)
def test_unparseable_python_fragments_still_flagged_by_regex_fallback(fragment: str) -> None:
    res = classify_code_chunk_security(fragment)
    assert res.is_insecure_pattern is True
    assert res.is_security_safeguard is False
    assert res.recommended_weight_multiplier == 0.25


@pytest.mark.parametrize(
    "snippet",
    [
        "import subprocess\ndef run(cmd, flag):\n    subprocess.run(cmd, shell=flag)\n",
        "import subprocess\ndef run(cmd):\n    subprocess.run(cmd, shell=1)\n",
        "import os\ndef run(cmd, args):\n    os.execv(cmd, args)\n",
        "import os\ndef run(cmd):\n    getattr(os, 'system')(cmd)\n",
        "import builtins\ndef run(expr):\n    builtins.eval(expr)\n",
        "from os import system as run_cmd\nrun_cmd(user_cmd)\n",
        'def query(cur, u):\n    q = f"SELECT * FROM users WHERE name = {u}"\n    cur.execute(q)\n',
    ],
)
def test_dynamic_shell_execv_getattr_builtins_eval_and_sql_caught(snippet: str) -> None:
    res = classify_code_chunk_security(snippet)
    assert res.is_insecure_pattern is True
    assert res.is_security_safeguard is False
    assert res.recommended_weight_multiplier == 0.25


@pytest.mark.parametrize(
    "safe_snippet",
    [
        'import subprocess\ndef run():\n    return subprocess.run(["ls", "-la"], shell=False)\n',
        'def submit_job(executor, job_id):\n    return executor.execute(f"worker-task-{job_id}")\n',
        'def submit_job_var(executor, job_id):\n    task_name = f"worker-task-{job_id}"\n    return executor.execute(task_name)\n',
    ],
)
def test_safe_subprocess_and_non_sql_execute_not_false_positived(safe_snippet: str) -> None:
    res = classify_code_chunk_security(safe_snippet)
    assert res.is_insecure_pattern is False
    assert res.recommended_weight_multiplier == 1.0


def test_backdoor_safeguard_with_arbitrary_call_does_not_get_boost() -> None:
    backdoor_stub = 'def validate_auth(token):\n    print("bypassed")\n    return True\n'
    res_bad = classify_code_chunk_security(
        backdoor_stub, symbol_name="validate_auth", award_safeguard_boost=True
    )
    assert res_bad.is_security_safeguard is False
    assert res_bad.recommended_weight_multiplier == 1.0

    res_default = classify_code_chunk_security(backdoor_stub, symbol_name="validate_auth")
    assert res_default.recommended_weight_multiplier == 1.0

    real_guard = (
        "import hmac\n"
        "def validate_auth(token: str, expected: str) -> bool:\n"
        "    if not hmac.compare_digest(token, expected):\n"
        "        raise PermissionError('invalid token')\n"
        "    return True\n"
    )
    res_good = classify_code_chunk_security(
        real_guard, symbol_name="validate_auth", award_safeguard_boost=True
    )
    assert res_good.is_insecure_pattern is False
    assert res_good.is_security_safeguard is True
    assert res_good.recommended_weight_multiplier == 1.35


@dataclass
class _MockHit:
    chunk_id: str | None
    chunk_text: str
    document_uri: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def test_quarantine_defaults_source_index_emission_and_origin_verified() -> None:
    hits = [
        _MockHit(
            chunk_id="poisoned-test",
            chunk_text="import os\nos.system(cmd)\n",
            document_uri="tests/test_backdoor.py",
            score=0.95,
            metadata={"origin_verified": False},
        ),
        _MockHit(
            chunk_id="safe-prod",
            chunk_text="def add(a, b):\n    return a + b\n",
            document_uri="src/app/math.py",
            score=0.50,
        ),
        _MockHit(
            chunk_id="verified-test",
            chunk_text="import os\nos.system(cmd)\n",
            document_uri="tests/test_verified.py",
            score=0.40,
            metadata={"origin_verified": True},
        ),
    ]

    reranked_default = rerank_chunks_for_case_bench(hits, quarantine_insecure=True)
    assert [r["chunk_id"] for r in reranked_default] == ["safe-prod", "verified-test"]
    assert reranked_default[0]["source_index"] == 1
    assert reranked_default[1]["source_index"] == 2

    reranked_keep_tests = rerank_chunks_for_case_bench(
        hits, quarantine_insecure=True, quarantine_tests=False
    )
    assert [r["chunk_id"] for r in reranked_keep_tests] == [
        "safe-prod",
        "verified-test",
        "poisoned-test",
    ]
    assert [r["source_index"] for r in reranked_keep_tests] == [1, 2, 0]


# --- Regression probes from test_regressions_from_review ---
@pytest.mark.parametrize(
    "snippet",
    [
        "else:\n    os.system(cmd)\n",
        "    except Exception:\n        data = pickle.loads(blob)\n",
        "import subprocess\nsubprocess.run(cmd, shell=1)\n",
        "import subprocess\nsubprocess.run(cmd, shell=flag)\n",
        "import os\ngetattr(os, 'system')(cmd)\n",
        "import builtins\nbuiltins.eval(x)\n",
        "import os\nos.execvp('sh', ['sh', '-c', cmd])\n",
        "x = os.popen(cmd)\nelse:\n  pass\n)",  # unparseable garbage still caught by fallback patterns
    ],
)
def test_g_previously_missed_insecure_patterns(snippet: str) -> None:
    assert classify_code_chunk_security(snippet).is_insecure_pattern


@pytest.mark.parametrize(
    "snippet",
    [
        "def q(cur, u):\n    cur.execute('SELECT * FROM t WHERE n=%s', (u,))\n",
        "def run(executor, name):\n    executor.execute(f'task-{name}')\n",
        "import subprocess\nsubprocess.run(['ls'], shell=False)\n",
    ],
)
def test_g_false_positives_removed(snippet: str) -> None:
    assert not classify_code_chunk_security(snippet).is_insecure_pattern


def test_g_sql_still_detected() -> None:
    assert classify_code_chunk_security(
        "def q(c,u):\n    c.execute(f\"SELECT * FROM t WHERE n='{u}'\")\n"
    ).is_insecure_pattern
    assert classify_code_chunk_security(
        "def q(c,u):\n    q = 'DELETE FROM t WHERE id=' + u\n    c.execute(q)\n"
    ).is_insecure_pattern


@pytest.mark.parametrize(
    "src,sym",
    [
        ("def validate_auth(t):\n    return bool(len(t))\n", "validate_auth"),
        ("def verify_signature(s, e):\n    print(s)\n    return True\n", "verify_signature"),
        ("def sanitize_input(x):\n    if x:\n        pass\n    return x\n", "sanitize_input"),
        (
            "import os\ndef authorize_admin(u):\n    return u == 'admin' or os.getenv('BYPASS') == '1'\n",
            "authorize_admin",
        ),
    ],
)
def test_g4_fake_safeguards_never_boosted(src: str, sym: str) -> None:
    r = classify_code_chunk_security(src, symbol_name=sym, award_safeguard_boost=True)
    assert r.recommended_weight_multiplier == 1.0


def test_g4_real_safeguard_boost_is_opt_in() -> None:
    src = "import hmac\ndef validate_auth(t, e):\n    if not hmac.compare_digest(t, e):\n        raise ValueError('bad')\n"
    assert (
        classify_code_chunk_security(src, symbol_name="validate_auth").recommended_weight_multiplier
        == 1.0
    )
    assert (
        classify_code_chunk_security(
            src, symbol_name="validate_auth", award_safeguard_boost=True
        ).recommended_weight_multiplier
        == 1.35
    )


def test_g5_tests_path_is_not_a_free_pass_without_verified_origin() -> None:
    hit = {
        "chunk_id": "a",
        "content": "import os\nos.system(c)\n",
        "source_uri": "tests/test_x.py",
        "score": 0.9,
    }
    assert rerank_chunks_for_case_bench([hit], quarantine_insecure=True) == []
    ok = dict(hit, metadata={"origin_verified": True})
    assert len(rerank_chunks_for_case_bench([ok], quarantine_insecure=True)) == 1


def test_p1_chunks_without_chunk_id_survive_and_map_back() -> None:
    hits = [{"content": "x = 1\n", "source_uri": "a.py", "score": 0.5}]
    out = rerank_chunks_for_case_bench(hits)
    assert out[0]["source_index"] == 0
