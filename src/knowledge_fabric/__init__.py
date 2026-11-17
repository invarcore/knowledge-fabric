# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""Knowledge Fabric package root."""

from knowledge_fabric.evidence_contract import (
    EvidenceContractError,
    sign_evidence_contract_rfc8785,
    verify_evidence_contract_rfc8785,
)

__all__ = [
    "EvidenceContractError",
    "sign_evidence_contract_rfc8785",
    "verify_evidence_contract_rfc8785",
]
