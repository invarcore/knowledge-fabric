# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Configuration loader for Knowledge Fabric runtime components."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml



@dataclass(slots=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass(slots=True)
class TikaSettings:
    endpoint: str
    request_timeout_seconds: int


@dataclass(slots=True)
class EmbeddingSettings:
    provider: str
    dimension: int
    batch_size: int


@dataclass(slots=True)
class RetrievalSettings:
    default_top_k: int
    lexical_weight: float
    vector_weight: float
    rrf_k: int
    backend: str = "postgres"


@dataclass(slots=True)
class RerankingSettings:
    provider: str = "none"
    top_n: int = 5


@dataclass(slots=True)
class AppSettings:
    name: str
    environment: str
    log_level: str


@dataclass(slots=True)
class Settings:
    app: AppSettings
    database: DatabaseSettings
    tika: TikaSettings
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings
    reranking: RerankingSettings


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    config_file = Path(path)
    if config_file.exists():
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    else:
        raw = {}

    app_raw = raw.get("app") or {}
    db_raw = raw.get("database") or {}
    tika_raw = raw.get("tika") or {}
    emb_raw = raw.get("embeddings") or {}
    retrieval_raw = raw.get("retrieval") or {}
    reranking_raw = raw.get("reranking") or {}

    # Database env overrides
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        parsed = urlparse(db_url)
        if parsed.hostname:
            db_raw["host"] = parsed.hostname
        if parsed.port:
            db_raw["port"] = parsed.port
        if parsed.path and parsed.path.lstrip("/"):
            db_raw["name"] = parsed.path.lstrip("/")
        if parsed.username:
            db_raw["user"] = parsed.username
        if parsed.password is not None:
            db_raw["password"] = parsed.password

    if os.environ.get("KF_DB_HOST") or os.environ.get("POSTGRES_HOST"):
        db_raw["host"] = os.environ.get("KF_DB_HOST") or os.environ.get("POSTGRES_HOST")
    if os.environ.get("KF_DB_PORT") or os.environ.get("POSTGRES_PORT"):
        db_raw["port"] = int(os.environ.get("KF_DB_PORT") or os.environ.get("POSTGRES_PORT") or 5432)
    if os.environ.get("KF_DB_NAME") or os.environ.get("POSTGRES_DB"):
        db_raw["name"] = os.environ.get("KF_DB_NAME") or os.environ.get("POSTGRES_DB")
    if os.environ.get("KF_DB_USER") or os.environ.get("POSTGRES_USER"):
        db_raw["user"] = os.environ.get("KF_DB_USER") or os.environ.get("POSTGRES_USER")
    if "KF_DB_PASSWORD" in os.environ or "POSTGRES_PASSWORD" in os.environ:
        db_raw["password"] = os.environ.get("KF_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD", "")

    # Tika env overrides
    if os.environ.get("TIKA_ENDPOINT"):
        tika_raw["endpoint"] = os.environ["TIKA_ENDPOINT"]

    # Embedding env overrides
    if os.environ.get("EMBEDDING_PROVIDER"):
        emb_raw["provider"] = os.environ["EMBEDDING_PROVIDER"]
    if os.environ.get("EMBEDDING_DIMENSION"):
        emb_raw["dimension"] = int(os.environ["EMBEDDING_DIMENSION"])

    # Retrieval env overrides
    if os.environ.get("RETRIEVAL_STORE_BACKEND") or os.environ.get("RETRIEVAL_BACKEND"):
        retrieval_raw["backend"] = os.environ.get("RETRIEVAL_STORE_BACKEND") or os.environ.get("RETRIEVAL_BACKEND")

    # Reranking env overrides
    if os.environ.get("RERANKER") or os.environ.get("RERANKING_PROVIDER"):
        reranking_raw["provider"] = os.environ.get("RERANKER") or os.environ.get("RERANKING_PROVIDER")

    return Settings(
        app=AppSettings(
            name=str(app_raw.get("name", "knowledge-fabric")),
            environment=str(app_raw.get("environment", "local")),
            log_level=str(app_raw.get("log_level", "INFO")),
        ),
        database=DatabaseSettings(
            host=str(db_raw.get("host", "localhost")),
            port=int(db_raw.get("port", 5432)),
            name=str(db_raw.get("name", "knowledge_fabric")),
            user=str(db_raw.get("user", "knowledge_fabric")),
            password=str(db_raw.get("password", "")),
        ),
        tika=TikaSettings(
            endpoint=str(tika_raw.get("endpoint", "http://localhost:9998")),
            request_timeout_seconds=int(tika_raw.get("request_timeout_seconds", 30)),
        ),
        embeddings=EmbeddingSettings(
            provider=str(emb_raw.get("provider", "mock")),
            dimension=int(emb_raw.get("dimension", 1536)),
            batch_size=int(emb_raw.get("batch_size", 64)),
        ),
        retrieval=RetrievalSettings(
            default_top_k=int(retrieval_raw.get("default_top_k", 10)),
            lexical_weight=float(retrieval_raw.get("lexical_weight", 0.5)),
            vector_weight=float(retrieval_raw.get("vector_weight", 0.5)),
            rrf_k=int(retrieval_raw.get("rrf_k", 60)),
            backend=str(retrieval_raw.get("backend", "postgres")),
        ),
        reranking=RerankingSettings(
            provider=str(reranking_raw.get("provider", "none")),
            top_n=int(reranking_raw.get("top_n", 5)),
        ),
    )


def _as_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for '{key}'")
    return value
