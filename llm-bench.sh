#!/bin/bash
# Thin wrapper around the Python CLI.
# Usage: ./llm-bench.sh <command> configs/<framework>/<config>.yaml [options]

set -euo pipefail
exec uv run llm-bench "$@"
