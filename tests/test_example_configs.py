"""Verify every config in configs/ loads cleanly.

Catches regressions where an example config drifts out of sync with the
schema (e.g. a renamed field, an unknown key after a refactor).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.core.config import Config

_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = _REPO_ROOT / "configs"


def _list_yaml() -> list[Path]:
    if not _CONFIGS_DIR.exists():
        return []
    return sorted(_CONFIGS_DIR.rglob("*.yaml"))


@pytest.mark.parametrize(
    "config_file",
    _list_yaml(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_example_config_loads(config_file: Path):
    cfg = Config.from_yaml_file(config_file)
    assert cfg.project.type
    assert cfg.branches
    for branch in cfg.branches.values():
        assert branch.runs
        for r in branch.runs:
            assert r.command  # post_init enforces, but double-check
