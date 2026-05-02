"""FrameworkAdapter protocol and registry.

An adapter is responsible for everything framework-specific:
  - which dataclass types parse the YAML's build/server sections
  - how to install the framework into a venv
  - how to describe its server (if any) — the runner manages lifecycle
  - how to clear framework-specific JIT caches
The core code (loader, runner, orchestrator) is framework-agnostic.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.config import BuildConfigBase, ServerConfigBase
from ..core.server import ServerSpec

if TYPE_CHECKING:
    from ..core.executor import BuildContext


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Per-framework knowledge plug-in.

    Implementations live in adapters/<framework>.py. Register via the
    `register()` function called at module top-level so the registration
    side effect fires on import.
    """

    # ── Identity ────────────────────────────────────────────────────
    name: str  # YAML's project.type matches this

    # ── Config types this adapter recognises ───────────────────────
    build_config_cls: type[BuildConfigBase]
    server_config_cls: type[ServerConfigBase] | None  # None = no server

    # ── Build ────────────────────────────────────────────────────────
    def build(
        self,
        repo_dir: Path,
        cfg: BuildConfigBase,
        max_jobs: int,
        ctx: BuildContext,
    ) -> None:
        """Install the framework into repo_dir/.venv. Called per-branch."""

    def build_identity(self, cfg: BuildConfigBase) -> dict:
        """Cache-key inputs: which fields, when changed, force a rebuild."""

    # ── Caches ──────────────────────────────────────────────────────
    def caches_to_clear(self) -> list[Path]:
        """Framework-specific JIT cache paths (cleared by `llm-bench clean`)."""

    # ── Server (optional) ───────────────────────────────────────────
    def server_spec(
        self,
        repo_dir: Path,
        cfg: ServerConfigBase,
    ) -> ServerSpec:
        """Build the server spec from this run's config. Only called when
        server_config_cls is not None and the run has a server config.
        Adapters with server_config_cls = None never receive this call."""


# ── Registry ─────────────────────────────────────────────────────────


_REGISTRY: dict[str, FrameworkAdapter] = {}


def register(adapter: FrameworkAdapter) -> FrameworkAdapter:
    """Register an adapter. Idempotent — re-registering the same name is fine."""
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_adapter(name: str) -> FrameworkAdapter:
    """Look up adapter by project.type. Lazy-imports built-in adapters on miss."""
    if name not in _REGISTRY:
        # Force-import known adapters so their register() side effects fire.
        # If an adapter file is absent (e.g. during partial bootstrap), fall
        # through to the "unknown type" error below.
        with contextlib.suppress(ImportError):
            from . import flash_attention, flashinfer, vllm  # noqa: F401
    if name not in _REGISTRY:
        known = sorted(_REGISTRY) or ["(none registered)"]
        raise ValueError(f"Unknown project.type {name!r}. Known: {known}")
    return _REGISTRY[name]


def known_adapters() -> list[str]:
    """List of currently registered project types."""
    return sorted(_REGISTRY)
