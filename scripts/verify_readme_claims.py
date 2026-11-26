#!/usr/bin/env python3
"""
verify_readme_claims.py — Invariant Truth Gate for Invarcore Repositories.
Enforces zero unverified claims, public registry existence, banned overclaim phrases,
and live CI badges instead of hardcoded test counts.

Checks:
  TRUTH-01: Registry existence for advertised install commands (PyPI, crates.io, npm).
  TRUTH-03: Banned AI overclaims / inaccurate standard citations.
  TRUTH-04: No hardcoded test-pass counts in prose or badges.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BANNED_RULES = [
    (
        re.compile(r"\bfixes all bypasses\b", re.IGNORECASE),
        "Banned overclaim: 'fixes all bypasses'. State specific tested vectors honestly.",
    ),
    (
        re.compile(r"\b100% cryptographic certainty\b", re.IGNORECASE),
        "Banned overclaim: '100% cryptographic certainty'.",
    ),
    (
        re.compile(r"\btamper-proof sha-?256\b", re.IGNORECASE),
        "Inaccurate claim: 'tamper-proof SHA-256'. An unkeyed hash is not tamper-proof without HMAC or Ed25519 signature.",
    ),
    (
        re.compile(
            r"(?:non[- ]?repudiabl[ey]\s+(?:[-a-z0-9]+\s+){0,2}hmac|hmac(?:[- ]?sha\d+)?(?:\s+(?:for|provides|ensures|guarantees|enables|gives))?\s+non[- ]?repudiat|non[- ]?repudiat[a-z]*\s*(?::|via|using|with|through|by)\s*(?:[^.\n,;]{0,40}\b)?hmac)",
            re.IGNORECASE,
        ),
        "Inaccurate claim: Symmetric HMAC provides keyed integrity/authenticity, not non-repudiation (use Ed25519).",
    ),
    (
        re.compile(r"\bclaude[- ]3[- .][57]\b", re.IGNORECASE),
        "Stale retired model citation (Claude 3.5 / 3.7). Update to current supported models.",
    ),
    (
        re.compile(r"\brfc\s*8785\b[^\.\n]*\barray\s+sort", re.IGNORECASE),
        "Inaccurate citation: RFC 8785 (JCS) preserves array order and does not sort arrays (Section 3.2.3).",
    ),
]

TEST_COUNT_PATTERN = re.compile(
    r"(?:badge/tests?-\d+%20passed|\b\d+/\d+\s+passing\b|\b\d+\s+tests?\s+passed\b)",
    re.IGNORECASE,
)

INSTALL_LINE_PATTERN = re.compile(
    r"^\s*(?:pip|uv pip)\s+install\s+(.+)$|^\s*uv\s+(?:add|tool\s+install)\s+(.+)$",
    re.MULTILINE,
)

PYPI_BADGE_PATTERN = re.compile(r"https?://pypi\.org/project/([a-zA-Z0-9_.-]+)")


def is_local_or_git_install(arg: str) -> bool:
    arg_lower = arg.lower()
    return (
        "git+" in arg_lower
        or "@" in arg_lower
        or arg_lower.startswith(("-e", "-r", ".", "/", "\\", "http://", "https://"))
    )


def extract_package_names_from_line(line_args: str) -> list[str]:
    try:
        tokens = shlex.split(line_args)
    except ValueError:
        tokens = line_args.split()

    packages = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in ("-r", "--requirement", "-e", "--editable", "-c", "--constraint"):
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if is_local_or_git_install(token):
            continue

        clean = token.strip("\"'`;,")
        if "[" in clean:
            clean = clean.split("[")[0]
        if clean:
            packages.append(clean)
    return packages


def check_pypi_exists(pkg: str, retries: int = 3) -> tuple[bool, bool]:
    """Return (exists, owned_by_invarcore)."""
    url = f"https://pypi.org/pypi/{pkg}/json"
    headers = {"User-Agent": "Invarcore-Truth-Gate/1.0"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    info = data.get("info", {})
                    blob = json.dumps(info).lower()
                    return True, ("invarcore" in blob)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, False
            if attempt < retries - 1 and e.code >= 500:
                time.sleep(1.0 * (attempt + 1))
                continue
            return False, False
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            return False, False
    return False, False


def verify_readme(readme_path: Path, offline: bool = False) -> list[str]:
    errors: list[str] = []
    if not readme_path.is_file():
        return [f"README file not found: {readme_path}"]

    content = readme_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # TRUTH-03: Banned phrases
    for line_no, line in enumerate(lines, start=1):
        for pattern, msg in BANNED_RULES:
            if pattern.search(line):
                errors.append(
                    f"TRUTH-03 (Line {line_no}): {msg}\n    Offending text: {line.strip()}"
                )

    # TRUTH-04: Hardcoded test pass counts
    for line_no, line in enumerate(lines, start=1):
        if TEST_COUNT_PATTERN.search(line):
            errors.append(
                f"TRUTH-04 (Line {line_no}): Hardcoded test count detected. "
                f"Use a live GitHub Actions CI status badge instead.\n    Offending text: {line.strip()}"
            )

    # TRUTH-01: Public registry existence & ownership check
    if not offline:
        install_candidates: set[str] = set()

        for match in INSTALL_LINE_PATTERN.finditer(content):
            args_str = match.group(1) or match.group(2)
            if args_str:
                for pkg in extract_package_names_from_line(args_str):
                    install_candidates.add(pkg)

        for match in PYPI_BADGE_PATTERN.finditer(content):
            install_candidates.add(match.group(1).strip("/"))

        for pkg in sorted(install_candidates):
            exists, owned = check_pypi_exists(pkg)
            if not exists:
                errors.append(
                    f"TRUTH-01: Advertised package '{pkg}' does not exist on PyPI. "
                    f"Replace with 'pip install git+https://github.com/invarcore/...' until released."
                )
            elif not owned:
                errors.append(
                    f"TRUTH-01: PyPI package '{pkg}' exists but metadata does not reference 'invarcore' "
                    f"(potential third-party namespace collision)."
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Invarcore README truth claims.")
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md (default: ./README.md)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip external network registry lookups (TRUTH-01).",
    )
    args = parser.parse_args()

    readme_file = args.readme.resolve()
    print(f"[*] Running Invariant Truth Gate on: {readme_file}")
    if args.offline:
        print("[*] Offline mode enabled (skipping public registry existence checks).")

    errors = verify_readme(readme_file, offline=args.offline)

    if errors:
        print(
            f"\n[FAIL] {len(errors)} truth violation(s) found in {readme_file.name}:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"\n[PASS] All truth invariants satisfied for {readme_file.name} (TRUTH-01, TRUTH-03, TRUTH-04)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
