# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Database access and schema bootstrap."""

from knowledge_fabric.db.audit import AuditLogger
from knowledge_fabric.db.connection import create_postgres_connection_factory
from knowledge_fabric.db.dimension_guard import DimensionMismatchError, check_embedding_dimension
from knowledge_fabric.db.repository import KnowledgeRepository

__all__ = [
    "AuditLogger",
    "DimensionMismatchError",
    "KnowledgeRepository",
    "check_embedding_dimension",
    "create_postgres_connection_factory",
]
