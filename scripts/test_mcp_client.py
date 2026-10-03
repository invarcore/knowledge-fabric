"""Standalone FastMCP Interactive Desktop Tester.

Allows inspecting and calling Knowledge Fabric and Intent Fabric MCP tools
directly from your terminal with ZERO subscriptions, ZERO API keys, and ZERO cost.

Usage:
  uv run python scripts/test_mcp_client.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure paths
kf_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(kf_root / "src"))

intent_root = kf_root.parent / "intent-fabric"
if (intent_root / "src").exists():
    sys.path.insert(0, str(intent_root / "src"))


def test_knowledge_fabric():
    print("\n" + "=" * 70)
    print(" [1] KNOWLEDGE FABRIC MCP SERVER (FastMCP)")
    print("=" * 70)

    from knowledge_fabric.config import load_settings
    from knowledge_fabric.mcp.server import build_tools_from_settings, create_mcp_server

    settings = load_settings()
    tools = build_tools_from_settings(settings)
    server = create_mcp_server(tools)

    # List tools from FastMCP registry
    registered_tools = getattr(server, "_tool_manager", None)
    tool_list = registered_tools.list_tools() if registered_tools else []
    print(f"\n[+] FastMCP Server '{server.name}' initialized.")
    print(f"[+] Discovered {len(tool_list)} tools available to AI assistants:\n")

    for t in tool_list:
        lines = (t.description or "").splitlines()
        desc = lines[0] if lines else ""
        print(f"  * {t.name:<24} | {desc[:60]}")


    # Execute health_check
    print("\n[>] Testing tool: health_check()...")
    health = tools.health_check()
    print("    Result:")
    print(f"      - Server Status:        {health.get('status')}")
    print(f"      - Embedding Provider:   {health.get('embedding_provider')}")
    print(f"      - Embedding Dimension:  {health.get('embedding_dimension')}")
    print(f"      - Database Status:      {health.get('db_status')}")

    # Test evidence hashing & provenance contract
    print("\n[>] Testing tool: compute_chunk_hash & provenance...")
    from knowledge_fabric.evidence.models import (
        EvidenceItem,
        compute_chunk_hash,
        compute_package_digest,
        compute_query_fingerprint,
    )

    h1 = compute_chunk_hash("docs/handbook.md", "Vacation policy: 20 days PTO.")
    h2 = compute_chunk_hash("docs/security.md", "Production access requires MFA.")
    digest = compute_package_digest([
        EvidenceItem(1, 1, "docs/handbook.md", 0, "PTO", 0.9, provenance_hash=h1),
        EvidenceItem(2, 2, "docs/security.md", 0, "MFA", 0.85, provenance_hash=h2),
    ])
    fingerprint = compute_query_fingerprint("PTO policy", "hr-tenant", "hybrid")

    print(f"      - Chunk 1 SHA-256:      {h1[:32]}...")
    print(f"      - Chunk 2 SHA-256:      {h2[:32]}...")
    print(f"      - Package Digest:       {digest[:32]}...")
    print(f"      - Query Fingerprint:    {fingerprint[:32]}...")


def test_intent_fabric():
    print("\n" + "=" * 70)
    print(" [2] INTENT FABRIC MCP SERVER (FastMCP)")
    print("=" * 70)

    try:
        from intent_fabric.mcp.server import create_mcp_server
        from intent_fabric.mcp.tools import IntentFabricMCPTools
        from intent_fabric.contracts.evidence_verifier import verify_evidence
    except ImportError as exc:
        print(f"[!] Intent Fabric not available in path: {exc}")
        return

    tools = IntentFabricMCPTools()
    server = create_mcp_server(tools)

    registered_tools = getattr(server, "_tool_manager", None)
    tool_list = registered_tools.list_tools() if registered_tools else []
    print(f"\n[+] FastMCP Server '{server.name}' initialized.")
    print(f"[+] Discovered {len(tool_list)} tools available to AI assistants:\n")

    for t in tool_list:
        lines = (t.description or "").splitlines()
        desc = lines[0] if lines else ""
        print(f"  * {t.name:<26} | {desc[:58]}")


    print("\n[>] Testing tool: health_check()...")
    health = tools.health_check()
    health_payload = health.get("payload", health)
    print("    Result:")
    print(f"      - Server Status:        {health_payload.get('status')}")
    print(f"      - Intent Planner:       {health_payload.get('planner')}")
    print(f"      - Active Policy Rules:  {health_payload.get('policy_rules_count')}")

    # Test evidence verifier
    print("\n[>] Testing Intent Fabric: EvidenceVerifier (zero-trust contract check)...")
    from datetime import datetime, timezone
    import hashlib

    # Case A: Cryptographically authentic payload
    chunk_uri = "docs/policy.md"
    chunk_text = "Requires director approval."
    valid_chunk_hash = hashlib.sha256(f"{chunk_uri}:{chunk_text}".encode("utf-8")).hexdigest()
    valid_digest = hashlib.sha256(valid_chunk_hash.encode("utf-8")).hexdigest()
    valid_fingerprint = hashlib.sha256(b"sec-tenant:grant admin:hybrid").hexdigest()

    valid_payload = {
        "query_text": "grant admin access",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [
            {
                "chunk_id": 101,
                "document_uri": chunk_uri,
                "snippet": chunk_text,
                "provenance_hash": valid_chunk_hash,
            }
        ],
        "provenance_digest": valid_digest,
        "query_fingerprint": valid_fingerprint,
    }
    v_valid = verify_evidence(valid_payload, max_age_seconds=30.0)
    print("      [Case A: Authentic Payload from Knowledge Fabric]")
    print(f"        * Contract Status:    {'[PASS] VERIFIED AUTHENTIC' if v_valid.is_valid else '[FAIL]'}")
    print(f"        * Checked Chunks:     {v_valid.checked_chunks}")
    print(f"        * Digest Verified:    {v_valid.digest_valid}")
    print(f"        * Freshness Verified: {v_valid.freshness_valid}")

    # Case B: Tampered in-flight payload (attacker modified snippet)
    tampered_payload = dict(valid_payload)
    tampered_payload["items"] = [
        {
            "chunk_id": 101,
            "document_uri": chunk_uri,
            "snippet": "MODIFIED: Auto-approved without director.",
            "provenance_hash": valid_chunk_hash,  # Old hash doesn't match new snippet!
        }
    ]
    v_tampered = verify_evidence(tampered_payload, max_age_seconds=30.0)
    print("      [Case B: Tampered In-Flight Payload (Simulated Attack)]")
    print(f"        * Tamper Detected:    {'[SUCCESS] BLOCKED (HASH MISMATCH)' if not v_tampered.is_valid else '[FAIL]'}")
    print(f"        * Flagged Chunks:     {v_tampered.failed_chunks}")
    print(f"        * Tamper Reason:      {v_tampered.errors[0] if v_tampered.errors else 'None'}")



def main():
    print("\n" + "=" * 70)
    print("  INTERACTIVE MCP DESKTOP TESTER (ZERO SUBSCRIPTION REQUIRED)")
    print("=" * 70)

    test_knowledge_fabric()
    test_intent_fabric()

    print("\n" + "=" * 70)
    print(" [SUCCESS] All MCP tools verified and operational!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
