"""Build-skip cache: persist what was last built, decide when to rebuild.

The cache key is `(commit, adapter_identity)` — `adapter.build_identity(cfg)`
declares which fields of the adapter's BuildConfig affect the build artifact.
Sharing a venv across runs of the same branch is fine as long as those
identity fields haven't changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .git import head_commit

_STATE_FILE = ".llm_bench_build.json"


def _state_path(repo_dir: Path) -> Path:
    return repo_dir / _STATE_FILE


def read(repo_dir: Path) -> dict:
    """Read the persisted build state, or {} if none."""
    p = _state_path(repo_dir)
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def write(repo_dir: Path, state: dict) -> None:
    """Persist the build state for future skip decisions."""
    with _state_path(repo_dir).open("w") as f:
        json.dump(state, f, indent=2)


def current_state(repo_dir: Path, adapter_identity: dict) -> dict:
    """Compute the current build state: HEAD commit + adapter's identity fields."""
    return {"commit": head_commit(repo_dir), **adapter_identity}


def should_skip(repo_dir: Path, current: dict) -> bool:
    """Return True if the persisted state matches `current` and FORCE_BUILD is unset."""
    if os.environ.get("FORCE_BUILD") == "1":
        return False
    return read(repo_dir) == current
