# Security Policy

## Supported Versions

We release patches and security fixes for the active minor release branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Security is paramount to Knowledge Fabric, especially given its role in multi-tenant data isolation and enterprise evidence retrieval.

If you believe you have discovered a security vulnerability in Knowledge Fabric:

1. **Do not open a public GitHub issue.**
2. Please submit a confidential report via GitHub Private Vulnerability Reporting or email the maintainers at `security@knowledge-fabric.dev`.
3. Provide a clear description, proof of concept, and reproduction steps.

We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline.

---

## Deployment & Security Configuration Flags

Knowledge Fabric provides configurable security guardrails that adapt to different enterprise workloads without disrupting high-throughput pipelines or large document processing:

| Environment Variable | Default | Purpose & Flexibility |
| :--- | :--- | :--- |
| `KNOWLEDGE_MAX_QUERY_LENGTH` | `4000` | Maximum length in characters for incoming query strings. Can be increased for deep analytical queries or error stack traces. |
| `KNOWLEDGE_MAX_TOP_K` | `100` | Maximum `top_k` results returned by lexical/vector search. Prevents unbounded payload sizes and out-of-memory errors. |
| `KNOWLEDGE_MAX_RERANK_CANDIDATES` | `100` | Maximum candidate pool evaluated by cross-encoder rerankers before returning top hits. Protects CPU/GPU from compute spikes. |
| `KNOWLEDGE_STRICT_TENANT_CHECK` | `true` | Enforces safe tenant namespace syntax (`^[a-zA-Z0-9_.:@-]{1,128}$`). Set to `false` if custom internal naming schemes are required. |
| `KNOWLEDGE_ENFORCE_NON_SUPERUSER` | `true` | Restricts database connection roles to dedicated least-privilege non-superusers (e.g. `kf_app`) to enforce PostgreSQL Row-Level Security (KF-01). Set to `false` to permit superusers for migrations/admin tooling. |

