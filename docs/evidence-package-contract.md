# Evidence Package Contract

Knowledge Fabric returns retrieval output as a structured evidence package.

## Purpose

- Provide transparent retrieval context to downstream consumers
- Separate retrieval outputs from answer generation
- Preserve source traceability

## Contract shape

```text
EvidencePackage
  - query_text: string
  - generated_at: ISO timestamp
  - items: EvidenceItem[]
  - retrieval_summary: object
```

```text
EvidenceItem
  - chunk_id: integer
  - document_id: integer
  - document_uri: string
  - chunk_index: integer
  - snippet: string
  - score: number
  - retrieval_sources: string[]
  - metadata: object
```

## Example response

```json
{
  "query_text": "vector retrieval with pgvector",
  "generated_at": "2026-07-12T08:00:00+00:00",
  "items": [
    {
      "chunk_id": 101,
      "document_id": 12,
      "document_uri": "sources/guide.md",
      "chunk_index": 3,
      "snippet": "Vector search uses embedding similarity...",
      "score": 0.124,
      "retrieval_sources": ["lexical", "vector"],
      "metadata": {
        "heading": "Vector Retrieval"
      }
    }
  ],
  "retrieval_summary": {
    "total_items": 1,
    "sources": ["lexical", "vector"]
  }
}
```

## Presentation Formats

To optimize token efficiency for autonomous AI agents, `retrieve_evidence` supports three presentation formats via the `format` parameter:

1. **`format="json"` (Default)**: Full JSON payload including provenance digest, query fingerprint, HMAC signature, and per-leg diagnostic summaries.
2. **`format="compact"`**: Lean, deduplicated JSON omitting backward-compatibility aliases and redundant text arrays, reducing wire footprint by ~50%.
3. **`format="markdown"`**: Returns pre-formatted structured markdown citations (`citations` field) ready for immediate agent prompting, reducing token overhead by >80%.

## Contract guarantees

- `retrieve_evidence` returns evidence, not final natural-language answers.
- `retrieval_sources` indicates which retrieval paths contributed to each item.
- `document_uri` and `chunk_id` enable deterministic trace-back.
