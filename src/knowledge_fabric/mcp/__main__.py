# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for `python -m knowledge_fabric.mcp`."""

from knowledge_fabric.mcp.server import run_mcp_server


if __name__ == "__main__":
    run_mcp_server()
