# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""CASE-Bench Code-Generation RAG Security Guard.

Addresses arXiv:2611.06294 (CASE-Bench / A.S.E 2.0) by classifying repository
code chunks with is_security_safeguard (boosting verified validation/authz/sanitization
helpers by 1.35x in RRF) and has_insecure_exemplar (quarantining or down-ranking
chunks containing eval, exec, shell=True, verify=False, unsafe yaml.load/unsafe_load,
subprocess.getoutput, asyncio.create_subprocess_shell, or dynamic SQL execution).
"""

from __future__ import annotations

import ast
import operator
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

SAFEGUARD_NAME_PATTERNS: Tuple[str, ...] = (
    "sanitize",
    "validate",
    "verify",
    "check_auth",
    "require_auth",
    "login_required",
    "permission_required",
    "requires_scope",
    "require_role",
    "require_permission",
    "enforce_policy",
    "check_bounds",
    "escape_html",
    "parameterize",
    "csrf_protect",
    "rate_limit",
    "authorize",
)

INSECURE_CALL_NAMES: Tuple[str, ...] = (
    "eval",
    "exec",
    "builtins.eval",
    "builtins.exec",
    "os.system",
    "os.popen",
    "pickle.loads",
    "pickle.load",
    "marshal.loads",
    "marshal.load",
    "tempfile.mktemp",
    "yaml.unsafe_load",
    "yaml.unsafe_load_all",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "asyncio.create_subprocess_shell",
)

_WEAK_HASH_CALL_NAMES: Tuple[str, ...] = ("hashlib.md5", "hashlib.sha1")

_SAFE_CALL_NAMES: Tuple[str, ...] = (
    "hmac.compare_digest",
    "secrets.compare_digest",
    "html.escape",
    "shlex.quote",
)

_DANGEROUS_STAR_IMPORTS: Set[str] = {"os", "subprocess", "yaml", "pickle", "marshal"}
_SQL_EXEC_METHODS: Tuple[str, ...] = ("execute", "executemany", "executescript")

_NON_PY_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bshell\s*=\s*True\b"), "kwarg:shell=True"),
    (re.compile(r"\bverify\s*=\s*False\b"), "kwarg:verify=False"),
    (re.compile(r"\beval\s*\("), "non_py:eval"),
    (re.compile(r"\bnew\s+Function\s*\("), "non_py:new_function"),
    (re.compile(r"\bchild_process\.(?:exec|execSync)\s*\("), "non_py:child_process.exec"),
    (re.compile(r"\binnerHTML\s*="), "non_py:dom_xss_innerhtml"),
    (re.compile(r"\bdangerouslySetInnerHTML\b"), "non_py:dangerouslySetInnerHTML"),
    (
        re.compile(r"`[^`]*\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^`]*\$\{", re.IGNORECASE),
        "non_py:sql_template_injection",
    ),
)


@dataclass(frozen=True)
class AnnotatedCodeChunk:
    """Retrieved code chunk annotated with CASE-Bench safeguard/insecure classification."""

    chunk_id: str
    source_uri: str
    symbol_name: str
    content: str
    base_rrf_score: float
    is_security_safeguard: bool
    has_insecure_exemplar: bool
    safeguard_signals: Tuple[str, ...]
    insecure_signals: Tuple[str, ...]
    adjusted_score: float


def dotted_name(node: ast.AST) -> str:
    """Resolve an ast.Name or ast.Attribute chain into a dotted identifier string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _resolve_alias(raw_name: str, alias_map: Mapping[str, str]) -> str:
    """Resolve leading imported identifier against module/symbol alias map."""
    if not raw_name:
        return ""
    parts = raw_name.split(".")
    if parts[0] in alias_map:
        resolved = alias_map[parts[0]]
        if operator.gt(len(parts), 1):
            return f"{resolved}.{'.'.join(parts[1:])}"
        return resolved
    return raw_name


def _build_alias_map(tree: ast.AST) -> Tuple[Dict[str, str], Set[str]]:
    """Collect import alias mappings and wildcard-imported insecure modules."""
    alias_map: Dict[str, str] = {}
    star_warnings: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias_map[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            root_mod = mod_name.split(".")[0]
            for alias in node.names:
                if alias.name == "*":
                    if root_mod in _DANGEROUS_STAR_IMPORTS:
                        star_warnings.add(f"import:wildcard_{root_mod}")
                    continue
                bound_name = alias.asname or alias.name
                full_target = f"{mod_name}.{alias.name}" if mod_name else alias.name
                alias_map[bound_name] = full_target
    return alias_map, star_warnings


def _is_dynamic_sql_expr(expr: ast.AST) -> Optional[str]:
    """Identify dynamic SQL construction expressions; returns tag or None."""
    if isinstance(expr, ast.JoinedStr):
        return "sql:fstring_injection"
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Mod, ast.Add)):
        return "sql:string_concat_injection"
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if expr.func.attr == "format":
            return "sql:format_injection"
    return None


def _is_substantive_function(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Reject trivial pass/constant-return stubs from safeguard promotion."""
    for child in ast.walk(fn_node):
        if isinstance(child, (ast.If, ast.Raise, ast.Assert, ast.Compare, ast.Call, ast.Try)):
            return True
    return False


def _has_yaml_safe_loader(node: ast.Call, alias_map: Mapping[str, str]) -> bool:
    for kw in node.keywords:
        if kw.arg == "Loader" and "Safe" in _resolve_alias(dotted_name(kw.value), alias_map):
            return True
    if not operator.lt(len(node.args), 2) and "Safe" in _resolve_alias(
        dotted_name(node.args[1]), alias_map
    ):
        return True
    return False


def _extract_sql_arg(node: ast.Call) -> Optional[ast.AST]:
    if node.args:
        return node.args[0]
    for kw in node.keywords:
        if kw.arg in ("sql", "statement", "query"):
            return kw.value
    return None


def _scan_non_python_fallback(source: str, insecure_hits: Set[str]) -> None:
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        for pat, label in _NON_PY_PATTERNS:
            if pat.search(stripped):
                insecure_hits.add(label)


def classify_code_chunk_security(
    source_code: str, symbol_name: str = ""
) -> Tuple[bool, bool, Tuple[str, ...], Tuple[str, ...]]:
    """Classify Python code chunk for verified safeguard signals and insecure exemplar smells."""
    safeguard_hits: Set[str] = set()
    insecure_hits: Set[str] = set()

    lower_sym = symbol_name.lower()
    for pat in SAFEGUARD_NAME_PATTERNS:
        if pat in lower_sym:
            safeguard_hits.add(f"symbol:{pat}")

    dedented = textwrap.dedent(source_code).strip()
    parsed_ok = False
    has_substantive_logic = False

    try:
        tree = ast.parse(dedented)
        parsed_ok = True
        alias_map, star_warnings = _build_alias_map(tree)
        insecure_hits.update(star_warnings)

        tainted_vars: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is not None:
                tag = _is_dynamic_sql_expr(node.value)
                if tag:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            tainted_vars[tgt.id] = tag
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                tag = _is_dynamic_sql_expr(node.value)
                if tag and isinstance(node.target, ast.Name):
                    tainted_vars[node.target.id] = tag

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_lower = node.name.lower()
                is_substantive = _is_substantive_function(node)
                for pat in SAFEGUARD_NAME_PATTERNS:
                    if pat in fn_lower:
                        safeguard_hits.add(f"fn_def:{pat}")
                        if is_substantive:
                            has_substantive_logic = True
                for dec in node.decorator_list:
                    dec_target = dec.func if isinstance(dec, ast.Call) else dec
                    dec_name = _resolve_alias(dotted_name(dec_target), alias_map).lower()
                    for pat in SAFEGUARD_NAME_PATTERNS:
                        if pat in dec_name:
                            safeguard_hits.add(f"decorator:{pat}")
                            if is_substantive:
                                has_substantive_logic = True

            elif isinstance(node, ast.Call):
                raw_name = dotted_name(node.func)
                call_name = _resolve_alias(raw_name, alias_map)
                if call_name in INSECURE_CALL_NAMES:
                    insecure_hits.add(f"call:{call_name}")

                if call_name in _WEAK_HASH_CALL_NAMES:
                    non_sec = any(
                        kw.arg == "usedforsecurity"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False
                        for kw in node.keywords
                    )
                    if not non_sec:
                        insecure_hits.add(f"call:{call_name}")

                if call_name in _SAFE_CALL_NAMES:
                    safeguard_hits.add(f"safe_call:{call_name}")
                    has_substantive_logic = True

                if call_name.startswith("subprocess.") or call_name in (
                    "subprocess.Popen",
                    "subprocess.run",
                    "subprocess.call",
                    "subprocess.check_output",
                    "Popen",
                    "run",
                    "call",
                    "check_output",
                ):
                    for kw in node.keywords:
                        if (
                            kw.arg == "shell"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            insecure_hits.add("kwarg:shell=True")

                if any(
                    call_name.startswith(p) for p in ("requests.", "httpx.", "session.", "client.")
                ):
                    for kw in node.keywords:
                        if (
                            kw.arg == "verify"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is False
                        ):
                            insecure_hits.add("kwarg:verify=False")

                if call_name in ("yaml.load", "yaml.load_all") and not _has_yaml_safe_loader(
                    node, alias_map
                ):
                    insecure_hits.add(f"call:{call_name}_unsafe")

                leaf = call_name.rsplit(".", 1)[-1]
                if leaf in _SQL_EXEC_METHODS:
                    sql_arg = _extract_sql_arg(node)
                    if sql_arg is not None:
                        tag = _is_dynamic_sql_expr(sql_arg)
                        if tag:
                            insecure_hits.add(tag)
                        elif isinstance(sql_arg, ast.Name) and sql_arg.id in tainted_vars:
                            insecure_hits.add(f"{tainted_vars[sql_arg.id]}:via_var")

        if any(s.startswith("symbol:") for s in safeguard_hits) and any(
            isinstance(n, (ast.If, ast.Raise, ast.Assert, ast.Call)) for n in ast.walk(tree)
        ):
            has_substantive_logic = True

    except SyntaxError:
        _scan_non_python_fallback(dedented, insecure_hits)

    is_safeguard = (
        parsed_ok and bool(safeguard_hits) and has_substantive_logic and not insecure_hits
    )
    has_insecure = bool(insecure_hits)
    return is_safeguard, has_insecure, tuple(sorted(safeguard_hits)), tuple(sorted(insecure_hits))


def _extract_chunk_fields(item: Any) -> Tuple[str, str, str, str, float]:
    """Normalize dict payloads as well as RetrievalHit, HybridHit, and Chunk objects."""
    if isinstance(item, Mapping):
        raw_meta = item.get("metadata")
        meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
        cid = str(item.get("chunk_id", item.get("chunk_index", "")))
        uri = str(item.get("source_uri") or item.get("document_uri") or "")
        sym = str(item.get("symbol_name") or meta.get("symbol_name") or "")
        raw_content = item.get("content")
        if raw_content is None:
            raw_content = item.get("chunk_text") or item.get("snippet") or ""
        content = str(raw_content)
        raw_score = item.get("base_rrf_score")
        if raw_score is None:
            raw_score = item.get("fused_score")
        if raw_score is None:
            raw_score = item.get("score")
        base_score = float(raw_score) if raw_score is not None else 0.0
        return cid, uri, sym, content, base_score

    inner = getattr(item, "hit", item)
    meta = getattr(inner, "metadata", {}) or {}
    cid = str(getattr(inner, "chunk_id", getattr(inner, "chunk_index", "")))
    uri = str(getattr(inner, "document_uri", ""))
    sym = str(meta.get("symbol_name", "")) if isinstance(meta, Mapping) else ""
    content = str(getattr(inner, "chunk_text", getattr(inner, "snippet", "")))
    base_score = float(getattr(item, "fused_score", getattr(inner, "score", 0.0)))
    return cid, uri, sym, content, base_score


def rerank_chunks_for_case_bench(
    chunks: Sequence[Any],
    *,
    safeguard_boost: float = 1.35,
    insecure_penalty: float = 0.15,
    quarantine_insecure: bool = True,
) -> List[AnnotatedCodeChunk]:
    """Rerank retrieved code chunks: boost safeguards and quarantine or penalize insecure exemplars."""
    annotated: List[AnnotatedCodeChunk] = []
    for item in chunks:
        cid, uri, sym, content, base_score = _extract_chunk_fields(item)
        is_safe, is_insec, safe_sigs, insec_sigs = classify_code_chunk_security(content, sym)
        if is_insec and quarantine_insecure:
            continue

        score = base_score
        if is_safe:
            score = round(base_score * safeguard_boost, 6)
        elif is_insec:
            score = round(base_score * insecure_penalty, 6)

        annotated.append(
            AnnotatedCodeChunk(
                chunk_id=cid,
                source_uri=uri,
                symbol_name=sym,
                content=content,
                base_rrf_score=base_score,
                is_security_safeguard=is_safe,
                has_insecure_exemplar=is_insec,
                safeguard_signals=safe_sigs,
                insecure_signals=insec_sigs,
                adjusted_score=score,
            )
        )

    annotated.sort(key=lambda c: (-c.adjusted_score, c.chunk_id))
    return annotated
