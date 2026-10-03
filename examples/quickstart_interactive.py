#!/usr/bin/env python3
"""
Knowledge Fabric - Interactive Quickstart Demo
Run directly with: python examples/quickstart_interactive.py

Demonstrates:
1. Lexical BM25 search matching exact codes & IDs
2. Dense Vector search matching conceptual semantics
3. Reciprocal Rank Fusion (RRF, k=60) combining both into a single ranked list
4. Sandboxed Evidence Package generation (<retrieved_evidence>)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DemoDoc:
    id: int
    title: str
    content: str
    tags: list[str]


# 1. Sample Enterprise Documents
SAMPLE_DOCS = [
    DemoDoc(
        id=101,
        title="AWS Aurora DB Failover Runbook [INC-8092]",
        content=(
            "In case of primary node failure, trigger automated cluster failover. "
            "Verification code INC-8092: verify replication lag is 0ms before promoting read replica."
        ),
        tags=["database", "incident", "aws", "runbook"],
    ),
    DemoDoc(
        id=102,
        title="HIPAA PHI Data Ingestion and Masking SOP",
        content=(
            "Patient identifiable information (PHI) must be scrubbed prior to model indexing. "
            "Any unmasked medical records or patient IDs must be quarantined immediately."
        ),
        tags=["compliance", "hipaa", "privacy", "security"],
    ),
    DemoDoc(
        id=103,
        title="SOC 2 KMS Key Rotation & Compromised Credential Protocol",
        content=(
            "When credentials or IAM secret keys are compromised, immediately revoke active sessions. "
            "Rotate AWS KMS customer managed keys and trigger an emergency audit log export."
        ),
        tags=["security", "soc2", "keys", "credentials"],
    ),
]


def run_demo():
    print("=" * 72)
    print("  KNOWLEDGE FABRIC - HYBRID RRF RETRIEVAL & EVIDENCE DEMO")
    print("=" * 72)
    print("\n[+] Initializing Knowledge Fabric hybrid pipeline...")
    print(f"[+] Loaded {len(SAMPLE_DOCS)} enterprise documents into mock index.")

    test_queries = [
        "INC-8092 failover steps",
        "how do we handle compromised database credentials",
    ]

    for query in test_queries:
        print("\n" + "-" * 72)
        print(f"QUERY: \"{query}\"")
        print("-" * 72)

        # 1. Lexical Scoring (Exact token overlap / BM25 simulation)
        query_terms = set(query.lower().split())
        lexical_scores = []
        for doc in SAMPLE_DOCS:
            doc_terms = set((doc.title + " " + doc.content).lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            lexical_scores.append((doc, overlap))
        lexical_ranked = sorted(lexical_scores, key=lambda x: x[1], reverse=True)

        # 2. Vector Semantic Scoring (Concept similarity simulation)
        vector_scores = []
        for doc in SAMPLE_DOCS:
            # Semantic weight simulation
            sim = 0.1
            if "credentials" in query.lower() and "compromised" in doc.content.lower():
                sim = 0.94
            elif "failover" in query.lower() and "failover" in doc.content.lower():
                sim = 0.91
            elif "patient" in query.lower() and "hipaa" in doc.tags:
                sim = 0.88
            vector_scores.append((doc, sim))
        vector_ranked = sorted(vector_scores, key=lambda x: x[1], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF: 1 / (k + rank)) with k=60
        k = 60
        fused_scores: dict[int, dict] = {}

        for rank, (doc, _) in enumerate(lexical_ranked, start=1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[doc.id] = {
                "doc": doc,
                "lex_rank": rank,
                "vec_rank": 0,
                "score": rrf_score,
                "sources": ["lexical"],
            }

        for rank, (doc, _) in enumerate(vector_ranked, start=1):
            rrf_score = 1.0 / (k + rank)
            if doc.id in fused_scores:
                fused_scores[doc.id]["score"] += rrf_score
                fused_scores[doc.id]["vec_rank"] = rank
                fused_scores[doc.id]["sources"].append("vector")
            else:
                fused_scores[doc.id] = {
                    "doc": doc,
                    "lex_rank": 0,
                    "vec_rank": rank,
                    "score": rrf_score,
                    "sources": ["vector"],
                }

        final_ranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

        print(f"\n{'Doc ID':<8} {'Fused RRF':<12} {'Lex Rank':<10} {'Vec Rank':<10} {'Title'}")
        print(f"{'-'*6:<8} {'-'*10:<12} {'-'*8:<10} {'-'*8:<10} {'-'*30}")
        for item in final_ranked:
            doc = item["doc"]
            print(
                f"{doc.id:<8} {item['score']:<12.5f} {item['lex_rank']:<10} {item['vec_rank']:<10} {doc.title[:38]}"
            )

        # 4. Output top result as Sandboxed Evidence XML
        top = final_ranked[0]
        top_doc = top["doc"]
        print("\n[+] Grounded Evidence Output for Autonomous Agent:")
        print("```xml")
        print("<retrieved_evidence>")
        print(f"  <item id=\"{top_doc.id}\" rrf_score=\"{top['score']:.4f}\" sources=\"{','.join(top['sources'])}\">")
        print(f"    <title>{top_doc.title}</title>")
        print(f"    <content>{top_doc.content}</content>")
        print("  </item>")
        print("</retrieved_evidence>")
        print("```")

    print("\n" + "=" * 72)
    print("SUCCESS: Hybrid RRF combines exact lexical codes with semantic understanding.")
    print("To connect Knowledge Fabric to your Claude Desktop or Cursor, run:")
    print("  knowledge-fabric-mcp")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_demo()
