# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""CASE-Bench Code-Generation RAG Security Guard (v3 Hardened).

Addresses arXiv:2611.06294 (CASE-Bench / A.S.E 2.0) by classifying repository
code chunks with is_security_safeguard (boosting verified validation/authz/sanitization
helpers by 1.35x in RRF when opt-in) and has_insecure_exemplar (quarantining or down-ranking
chunks containing eval, exec, truthy shell, verify=False, unsafe yaml.load/unsafe_load,
subprocess.getoutput, asyncio.create_subprocess_shell, or dynamic SQL execution).
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SAFEGUARD_NAME_PATTERNS: tuple[str, ...] = (
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

SAFEGUARD_PREFIXES: tuple[str, ...] = (
    "validate",
    "verify",
    "sanitize",
    "escape",
    "authorize",
    "authenticate",
    "enforce",
    "check_permission",
    "require_auth",
    "require_role",
    "csrf_protect",
    "rate_limit",
)

SAFEGUARD_DECORATORS: frozenset[str] = frozenset(
    {
        "require_auth",
        "login_required",
        "permission_required",
        "requires_scope",
        "csrf_protect",
        "validate_input",
        "enforce_policy",
        "require_role",
        "require_permission",
    }
)

VERIFYING_CALL_LEAVES: frozenset[str] = frozenset(
    {"compare_digest", "checkpw", "verify_signature", "check_password"}
)

INSECURE_CALL_NAMES: tuple[str, ...] = (
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

UNSAFE_CALLS_DIRECT: dict[str, tuple[str, str]] = {
    "eval": ("builtin:eval", "call:eval"),
    "exec": ("builtin:exec", "call:exec"),
    "__import__": ("builtin:dunder_import", "call:__import__"),
    "builtins.eval": ("builtin:eval", "call:builtins.eval"),
    "builtins.exec": ("builtin:exec", "call:builtins.exec"),
    "os.system": ("os:system", "call:os.system"),
    "os.popen": ("os:popen", "call:os.popen"),
    "subprocess.getoutput": ("subprocess:getoutput", "call:subprocess.getoutput"),
    "subprocess.getstatusoutput": ("subprocess:getstatusoutput", "call:subprocess.getstatusoutput"),
    "asyncio.create_subprocess_shell": (
        "asyncio:create_subprocess_shell",
        "call:asyncio.create_subprocess_shell",
    ),
    "pickle.loads": ("pickle:loads", "call:pickle.loads"),
    "pickle.load": ("pickle:load", "call:pickle.load"),
    "marshal.loads": ("marshal:loads", "call:marshal.loads"),
    "marshal.load": ("marshal:load", "call:marshal.load"),
    "yaml.unsafe_load": ("yaml:unsafe_load", "call:yaml.unsafe_load"),
    "yaml.unsafe_load_all": ("yaml:unsafe_load_all", "call:yaml.unsafe_load_all"),
    "tempfile.mktemp": ("tempfile:mktemp", "call:tempfile.mktemp"),
}

UNSAFE_CALL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("os.exec", "os:exec"),
    ("os.spawn", "os:spawn"),
)

UNSAFE_GETATTR_NAMES: frozenset[str] = frozenset({"system", "popen", "eval", "exec", "__import__"})

SUBPROCESS_FUNCS: frozenset[str] = frozenset({"run", "Popen", "call", "check_call", "check_output"})

SQL_EXECUTE_METHODS: frozenset[str] = frozenset({"execute", "executemany", "executescript", "raw"})

# G1/G6: SQL-ness is decided by the SHAPE of the string, not by the method name alone.
_SQL_SHAPE_RE = re.compile(
    r"\b(select|insert|update|delete|drop|alter|create|merge)\b.{0,300}?\b(from|into|set|table|where|values|join)\b",
    re.IGNORECASE | re.DOTALL,
)

NON_PYTHON_INSECURE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bshell\s*=\s*(?!False\b|0\b|None\b)\w"), "kwarg:shell=True"),
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
    (re.compile(r"\bos\.(system|popen|exec\w*|spawn\w*)\s*\("), "os:process_exec"),
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle:load"),
    (re.compile(r"\bmarshal\.loads?\s*\("), "marshal:load"),
    (re.compile(r"\byaml\.unsafe_load(_all)?\s*\("), "yaml:unsafe_load"),
    (
        re.compile(r"\bgetattr\s*\(\s*\w+\s*,\s*['\"](system|popen|eval|exec)['\"]\s*\)"),
        "builtin:getattr_unsafe",
    ),
    (re.compile(r"\bexec\s*\("), "builtin:exec"),
)

_TEST_OR_DOC_URI_RE = re.compile(
    r"(^|/)(tests?|docs?|examples?)/|(^|/)test_[^/]+\.py$|_test\.py$",
    re.IGNORECASE,
)

_FRAGMENT_PREFIX = {
    "else": "if _x:\n    pass\n",
    "elif": "if _x:\n    pass\n",
    "except": "try:\n    pass\n",
    "finally": "try:\n    pass\n",
}


@dataclass(frozen=True)
class CodeChunkSecurityClassification:
    """Security classification outcome for a code chunk."""

    is_insecure_pattern: bool
    is_security_safeguard: bool
    insecure_signals: tuple[str, ...]
    safeguard_signals: tuple[str, ...]
    recommended_weight_multiplier: float

    @property
    def has_insecure_exemplar(self) -> bool:
        return self.is_insecure_pattern

    def __iter__(self) -> Any:
        # Backward-compatible 4-tuple unpacking: (is_safe, is_insec, safe_sigs, insec_sigs)
        return iter(
            (
                self.is_security_safeguard,
                self.is_insecure_pattern,
                self.safeguard_signals,
                self.insecure_signals,
            )
        )


class AnnotatedCodeChunk(dict):
    """Retrieved code chunk annotated with CASE-Bench safeguard/insecure classification."""

    def __init__(
        self,
        *args: Any,
        chunk_id: str | None = None,
        source_uri: str = "",
        symbol_name: str = "",
        content: str = "",
        base_rrf_score: float = 0.0,
        is_security_safeguard: bool = False,
        has_insecure_exemplar: bool = False,
        safeguard_signals: tuple[str, ...] = (),
        insecure_signals: tuple[str, ...] = (),
        adjusted_score: float = 0.0,
        source_index: int = 0,
        case_guard: Any = None,
        document_uri: str = "",
        chunk_text: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if chunk_id is not None or "chunk_id" not in self:
            self["chunk_id"] = chunk_id
        if source_uri or "source_uri" not in self:
            self["source_uri"] = source_uri
        if document_uri or "document_uri" not in self:
            self["document_uri"] = document_uri or source_uri
        if symbol_name or "symbol_name" not in self:
            self["symbol_name"] = symbol_name
        if content or "content" not in self:
            self["content"] = content
        if chunk_text or "chunk_text" not in self:
            self["chunk_text"] = chunk_text or content
        if "base_rrf_score" not in self:
            self["base_rrf_score"] = base_rrf_score
        if "is_security_safeguard" not in self:
            self["is_security_safeguard"] = is_security_safeguard
        if "has_insecure_exemplar" not in self:
            self["has_insecure_exemplar"] = has_insecure_exemplar
        if "safeguard_signals" not in self:
            self["safeguard_signals"] = safeguard_signals
        if "insecure_signals" not in self:
            self["insecure_signals"] = insecure_signals
        if "adjusted_score" not in self:
            self["adjusted_score"] = adjusted_score
        if "source_index" not in self:
            self["source_index"] = source_index
        if case_guard is not None:
            self["case_guard"] = case_guard
        if metadata is not None:
            self["metadata"] = metadata

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AnnotatedCodeChunk' object has no attribute {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @property
    def is_insecure_pattern(self) -> bool:
        return bool(self.get("has_insecure_exemplar", self.get("is_insecure_pattern", False)))


def dotted_name(node: ast.AST) -> str:
    """Resolve an ast.Name or ast.Attribute chain into a dotted identifier string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _dotted_name_with_alias(node: ast.AST, alias_map: Mapping[str, str]) -> str | None:
    parts: list[str] = []
    curr = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(alias_map.get(curr.id, curr.id))
        return ".".join(reversed(parts))
    return None


def _const_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [
            v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
    if isinstance(node, ast.BinOp):
        return _const_parts(node.left) + _const_parts(node.right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _const_parts(node.func.value)
    return []


def _has_dynamic_part(node: ast.AST) -> bool:
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(v, ast.FormattedValue) and not isinstance(v.value, ast.Constant)
            for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return any(
            not isinstance(s, ast.Constant)
            and (not isinstance(s, ast.BinOp) or _has_dynamic_part(s))
            for s in (node.left, node.right)
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return any(
            not isinstance(a, ast.Constant)
            for a in list(node.args) + [k.value for k in node.keywords]
        )
    return False


def _is_dynamic_sql_expr(node: ast.AST, tainted_vars: frozenset[str]) -> tuple[bool, str | None]:
    if isinstance(node, ast.Name):
        if node.id in tainted_vars:
            return True, "via_var"
        return False, None

    has_dyn = _has_dynamic_part(node)
    const_text = " ".join(_const_parts(node))
    is_sql_shape = bool(_SQL_SHAPE_RE.search(const_text))
    if has_dyn and is_sql_shape:
        if isinstance(node, ast.JoinedStr):
            return True, "fstring"
        if isinstance(node, ast.BinOp):
            return True, "concat"
        if isinstance(node, ast.Call):
            return True, "format"
        return True, "dynamic"
    return False, None


def _has_check_evidence(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(fn_node):
        if isinstance(child, (ast.Raise, ast.Assert)):
            return True
        if isinstance(child, ast.Call):
            leaf = dotted_name(child.func).split(".")[-1]
            if leaf in VERIFYING_CALL_LEAVES:
                return True
    return False


def _matches_safeguard_name(name: str) -> bool:
    lower = name.lower().lstrip("_")
    return any(lower.startswith(p) for p in SAFEGUARD_PREFIXES)


def _shell_is_truthy(kw: ast.keyword) -> bool:
    """G2: shell=flag, shell=1, shell=cfg['x'] are all treated as unsafe; only constant False/0/None is safe."""
    return not (isinstance(kw.value, ast.Constant) and kw.value.value in (False, 0, None))


class _CaseGuardASTVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.insecure_signals: set[str] = set()
        self.safeguard_signals: set[str] = set()
        self.alias_map: dict[str, str] = {}
        self.tainted_sql_vars: set[str] = set()
        self.has_substantive_safeguard: bool = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            clean = alias.name.strip()
            self.alias_map[alias.asname or clean.split(".")[0]] = clean

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").strip()
        root_mod = mod.split(".")[0]
        for alias in node.names:
            if alias.name == "*":
                if root_mod in {"os", "subprocess", "yaml", "pickle", "marshal"}:
                    self.insecure_signals.add(f"import:wildcard_{root_mod}")
                continue
            bound_name = alias.asname or alias.name
            full_target = f"{mod}.{alias.name}" if mod else alias.name
            self.alias_map[bound_name] = full_target

    def visit_Assign(self, node: ast.Assign) -> None:
        if node.value is not None:
            is_sql, _ = _is_dynamic_sql_expr(node.value, frozenset(self.tainted_sql_vars))
            if is_sql:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.tainted_sql_vars.add(t.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            is_sql, _ = _is_dynamic_sql_expr(node.value, frozenset(self.tainted_sql_vars))
            if is_sql:
                self.tainted_sql_vars.add(node.target.id)
        self.generic_visit(node)

    def _visit_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            name = _dotted_name_with_alias(target, self.alias_map) or dotted_name(target)
            if name:
                leaf = name.split(".")[-1]
                for p in SAFEGUARD_NAME_PATTERNS:
                    if p in leaf.lower():
                        self.safeguard_signals.add(f"decorator:{p}")
                if leaf in SAFEGUARD_DECORATORS:
                    self.safeguard_signals.add(f"decorator:{leaf}")
                    self.has_substantive_safeguard = True

        if _matches_safeguard_name(node.name) and _has_check_evidence(node):
            self.safeguard_signals.add(f"fn_def:{node.name}")
            for p in SAFEGUARD_PREFIXES:
                if node.name.lower().startswith(p):
                    self.safeguard_signals.add(f"fn_def:{p}")
            self.has_substantive_safeguard = True

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_fn(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name_with_alias(node.func, self.alias_map) or dotted_name(node.func)
        if dotted:
            d = dotted.strip()
            leaf = d.split(".")[-1]

            if d in UNSAFE_CALLS_DIRECT:
                v3_label, legacy_label = UNSAFE_CALLS_DIRECT[d]
                self.insecure_signals.add(v3_label)
                self.insecure_signals.add(legacy_label)

            for prefix, label in UNSAFE_CALL_PREFIXES:
                if d.startswith(prefix):
                    self.insecure_signals.add(label)
                    self.insecure_signals.add("os:process_exec")

            if leaf in ("md5", "sha1") and d.startswith("hashlib."):
                non_sec = any(
                    kw.arg == "usedforsecurity"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                    for kw in node.keywords
                )
                if not non_sec:
                    self.insecure_signals.add(f"crypto:weak_{leaf}")
                    self.insecure_signals.add(f"call:hashlib.{leaf}")
            elif leaf == "mktemp" and "tempfile" in d:
                self.insecure_signals.add("tempfile:mktemp")
                self.insecure_signals.add("call:tempfile.mktemp")

            if (
                d.startswith("subprocess.")
                or d in ("run", "Popen", "call", "check_call", "check_output")
            ) and leaf in SUBPROCESS_FUNCS:
                for kw in node.keywords:
                    if kw.arg == "shell" and _shell_is_truthy(kw):
                        self.insecure_signals.add("subprocess:shell_truthy")
                        self.insecure_signals.add("kwarg:shell=True")

            if any(d.startswith(p) for p in ("requests.", "httpx.", "session.", "client.")):
                for kw in node.keywords:
                    if (
                        kw.arg == "verify"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False
                    ):
                        self.insecure_signals.add("tls:verify_false")
                        self.insecure_signals.add("kwarg:verify=False")

            if d in ("yaml.load", "yaml.load_all"):
                safe = False
                cands = list(node.args[1:2]) + [k.value for k in node.keywords if k.arg == "Loader"]
                for c in cands:
                    c_name = _dotted_name_with_alias(c, self.alias_map) or dotted_name(c)
                    if "Safe" in c_name:
                        safe = True
                if not safe:
                    self.insecure_signals.add(f"yaml:unverified_{leaf}")
                    self.insecure_signals.add(f"call:yaml.{leaf}_unsafe")

            if d in ("getattr", "builtins.getattr") and len(node.args) >= 2:
                a = node.args[1]
                if isinstance(a, ast.Constant) and a.value in UNSAFE_GETATTR_NAMES:
                    self.insecure_signals.add("builtin:getattr_unsafe_attr")
                    self.insecure_signals.add("builtin:getattr_unsafe")

            if leaf == "compare_digest":
                self.safeguard_signals.add("crypto:compare_digest")
                self.safeguard_signals.add("safe_call:hmac.compare_digest")
                self.has_substantive_safeguard = True
            elif leaf in VERIFYING_CALL_LEAVES:
                self.safeguard_signals.add(f"safe_call:{leaf}")
                self.has_substantive_safeguard = True

        for kw in node.keywords:
            if (
                kw.arg == "verify"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
            ):
                self.insecure_signals.add("tls:verify_false")
                self.insecure_signals.add("kwarg:verify=False")

        if isinstance(node.func, ast.Attribute) and node.func.attr in SQL_EXECUTE_METHODS:
            sql_arg = node.args[0] if node.args else None
            if sql_arg is None:
                for kw in node.keywords:
                    if kw.arg in ("sql", "statement", "query"):
                        sql_arg = kw.value
                        break
            if sql_arg is not None:
                is_dyn, kind = _is_dynamic_sql_expr(sql_arg, frozenset(self.tainted_sql_vars))
                if is_dyn:
                    self.insecure_signals.add("sql:dynamic_query_execution")
                    if kind == "fstring":
                        self.insecure_signals.add("sql:fstring_injection")
                    elif kind == "concat":
                        self.insecure_signals.add("sql:string_concat_injection")
                    elif kind == "format":
                        self.insecure_signals.add("sql:format_injection")
                    elif kind == "via_var":
                        self.insecure_signals.add("sql:fstring_injection:via_var")
                        self.insecure_signals.add("sql:string_concat_injection:via_var")

        self.generic_visit(node)


def _parse_python(dedented: str) -> ast.AST | None:
    """G1: try the chunk as-is, then repaired as a fragment (else:/except:/finally: openers)."""
    try:
        return ast.parse(dedented)
    except SyntaxError:
        pass

    for line in dedented.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            first_token = stripped.split(None, 1)[0].rstrip(":")
            prefix = _FRAGMENT_PREFIX.get(first_token)
            if prefix:
                try:
                    return ast.parse(prefix + dedented)
                except SyntaxError:
                    return None
            break
    return None


def classify_code_chunk_security(
    code_text: str,
    symbol_name: str | None = None,
    *,
    source_uri: str | None = None,
    award_safeguard_boost: bool = False,
) -> CodeChunkSecurityClassification:
    """Classify code chunk for verified safeguard signals and insecure exemplar smells.

    G4: names are attacker-controlled; boost is opt-in (award_safeguard_boost=False default).
    """
    dedented = textwrap.dedent(code_text or "").strip()
    if not dedented:
        return CodeChunkSecurityClassification(False, False, (), (), 1.0)

    tree = _parse_python(dedented)
    insecure: set[str] = set()
    safeguards: set[str] = set()

    if tree is not None:
        v = _CaseGuardASTVisitor()
        v.visit(tree)
        insecure.update(v.insecure_signals)
        safeguards.update(v.safeguard_signals)

        if symbol_name:
            lower_sym = symbol_name.lower()
            for p in SAFEGUARD_NAME_PATTERNS:
                if p in lower_sym:
                    if v.has_substantive_safeguard or _has_check_evidence_in_tree(tree):
                        safeguards.add(f"symbol:{p}")
            if _matches_safeguard_name(symbol_name) and (
                v.has_substantive_safeguard or _has_check_evidence_in_tree(tree)
            ):
                safeguards.add(f"symbol:{symbol_name}")
    else:
        # Unparseable: scan polyglot AND Python patterns; never earn a safeguard boost.
        for raw_line in dedented.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(("#", "//")):
                continue
            for pat, label in NON_PYTHON_INSECURE_PATTERNS:
                if pat.search(stripped):
                    insecure.add(label)

    is_insecure = bool(insecure)
    is_safeguard = (not is_insecure) and bool(safeguards)
    if is_insecure:
        mult = 0.25
    elif is_safeguard and award_safeguard_boost:
        mult = 1.35
    else:
        mult = 1.0

    return CodeChunkSecurityClassification(
        is_insecure,
        is_safeguard,
        tuple(sorted(insecure)),
        tuple(sorted(safeguards)) if not is_insecure else tuple(sorted(safeguards)),
        mult,
    )


def _has_check_evidence_in_tree(tree: ast.AST) -> bool:
    for child in ast.walk(tree):
        if isinstance(child, (ast.Raise, ast.Assert)):
            return True
        if isinstance(child, ast.Call):
            leaf = dotted_name(child.func).split(".")[-1]
            if leaf in VERIFYING_CALL_LEAVES:
                return True
    return False


def _extract_chunk_fields(
    item: Any,
) -> tuple[str | None, str, str | None, str, float, Mapping[str, Any]]:
    """Extract standard fields from dict payloads, Chunk, RetrievalHit, and HybridHit."""
    if isinstance(item, Mapping):
        raw_meta = item.get("metadata")
        meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
        cid = item.get("chunk_id")
        if cid is None:
            cid = item.get("chunk_index")
        cid_str = str(cid) if cid is not None else None
        uri = str(item.get("source_uri") or item.get("document_uri") or "")
        sym = item.get("symbol_name") or meta.get("symbol_name")
        sym_str = str(sym) if sym is not None else None
        content = str(item.get("content") or item.get("chunk_text") or item.get("snippet") or "")
        raw_score = item.get("base_rrf_score")
        if raw_score is None:
            raw_score = item.get("fused_score")
        if raw_score is None:
            raw_score = item.get("score", 0.0)
        base_score = float(raw_score) if raw_score is not None else 0.0
        return cid_str, uri, sym_str, content, base_score, meta

    inner = getattr(item, "hit", item)
    raw_meta = getattr(inner, "metadata", getattr(item, "metadata", {}))
    meta = raw_meta if isinstance(raw_meta, Mapping) else {}
    cid = getattr(inner, "chunk_id", getattr(inner, "chunk_index", getattr(item, "chunk_id", None)))
    cid_str = str(cid) if cid is not None else None
    uri = str(
        getattr(
            inner,
            "document_uri",
            getattr(
                inner, "source_uri", getattr(item, "document_uri", getattr(item, "source_uri", ""))
            ),
        )
    )
    sym = getattr(inner, "symbol_name", getattr(item, "symbol_name", meta.get("symbol_name")))
    sym_str = str(sym) if sym is not None else None
    content = str(
        getattr(
            inner,
            "chunk_text",
            getattr(
                inner,
                "content",
                getattr(
                    inner, "snippet", getattr(item, "chunk_text", getattr(item, "content", ""))
                ),
            ),
        )
    )
    raw_score = getattr(
        item, "base_rrf_score", getattr(item, "fused_score", getattr(inner, "score", 0.0))
    )
    base_score = float(raw_score) if raw_score is not None else 0.0
    return cid_str, uri, sym_str, content, base_score, meta


def rerank_chunks_for_case_bench(
    chunks: Sequence[Any],
    *,
    quarantine_insecure: bool = True,
    quarantine_tests: bool | None = None,
    award_safeguard_boost: bool = False,
    safeguard_boost: float = 1.35,
    insecure_penalty: float | None = None,
) -> list[AnnotatedCodeChunk]:
    """Rerank retrieved code chunks: boost safeguards and quarantine or penalize insecure exemplars."""
    reranked: list[AnnotatedCodeChunk] = []
    for index, item in enumerate(chunks):
        cid, uri, symbol, content, base_score, meta = _extract_chunk_fields(item)
        cls = classify_code_chunk_security(
            content,
            symbol_name=symbol,
            source_uri=uri,
            award_safeguard_boost=award_safeguard_boost,
        )

        # G5: a path in the URI is corpus-controlled; exempt only when ingestion verified the origin.
        is_test_path = bool(uri and _TEST_OR_DOC_URI_RE.search(uri.replace("\\", "/")))
        is_verified = meta.get("origin_verified") is True
        is_verified_test_or_doc = is_test_path and is_verified

        if quarantine_tests is False:
            is_exempt = is_test_path
        elif quarantine_tests is True:
            is_exempt = False
        else:
            is_exempt = is_verified_test_or_doc

        if cls.is_insecure_pattern and quarantine_insecure and not is_exempt:
            continue

        if isinstance(item, Mapping):
            entry_dict = dict(item)
        else:
            entry_dict = {
                "chunk_id": cid,
                "content": content,
                "chunk_text": content,
                "source_uri": uri,
                "document_uri": uri,
                "symbol_name": symbol,
                "base_rrf_score": base_score,
                "metadata": dict(meta),
            }

        if is_verified_test_or_doc:
            adjusted = round(base_score, 6)
        elif cls.is_insecure_pattern:
            pen = (
                insecure_penalty
                if insecure_penalty is not None
                else cls.recommended_weight_multiplier
            )
            adjusted = round(base_score * pen, 6)
        elif cls.is_security_safeguard and award_safeguard_boost:
            adjusted = round(base_score * safeguard_boost, 6)
        else:
            adjusted = round(base_score * cls.recommended_weight_multiplier, 6)

        entry_dict["source_index"] = index  # P1: lets callers map back WITHOUT relying on chunk_id
        entry_dict["case_guard"] = cls
        entry_dict["adjusted_score"] = adjusted
        entry_dict["is_security_safeguard"] = cls.is_security_safeguard
        entry_dict["has_insecure_exemplar"] = cls.is_insecure_pattern
        entry_dict["safeguard_signals"] = cls.safeguard_signals
        entry_dict["insecure_signals"] = cls.insecure_signals

        annotated = AnnotatedCodeChunk(
            entry_dict,
            chunk_id=cid,
            source_uri=uri,
            symbol_name=symbol or "",
            content=content,
            base_rrf_score=base_score,
            is_security_safeguard=cls.is_security_safeguard,
            has_insecure_exemplar=cls.is_insecure_pattern,
            safeguard_signals=cls.safeguard_signals,
            insecure_signals=cls.insecure_signals,
            adjusted_score=adjusted,
            source_index=index,
            case_guard=cls,
            document_uri=uri,
            chunk_text=content,
            metadata=dict(meta),
        )
        reranked.append(annotated)

    reranked.sort(key=lambda x: (-x["adjusted_score"], str(x.get("chunk_id") or "")))
    return reranked


__all__ = [
    "AnnotatedCodeChunk",
    "CodeChunkSecurityClassification",
    "INSECURE_CALL_NAMES",
    "SAFEGUARD_NAME_PATTERNS",
    "classify_code_chunk_security",
    "dotted_name",
    "rerank_chunks_for_case_bench",
]
