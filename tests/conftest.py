"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Write YAML to a tmp file and return the path."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        p = tmp_path / name
        p.write_text(content)
        return p

    return _write
