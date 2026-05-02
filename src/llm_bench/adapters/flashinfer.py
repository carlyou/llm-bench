"""FlashInfer adapter.

Encodes how to install FlashInfer into a venv as a library (no server).
Build is one editable install with --no-build-isolation, plus a torch
install first and a submodule init for cutlass / spdlog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.build import (
    init_submodules,
    install_torch_cuda,
    passthrough_build_env,
    setup_compiler_cache_env,
    venv_python,
)
from ..core.config import BuildConfigBase, ServerConfigBase
from ..core.executor import BuildContext, run
from .base import register

# ── Configs ──────────────────────────────────────────────────────────


@dataclass
class FlashinferBuildConfig(BuildConfigBase):
    """flashinfer-specific build knobs."""

    nvcc_threads: int = 4  # FLASHINFER_NVCC_THREADS
    torch_index: str = "https://download.pytorch.org/whl/cu130"


# No FlashinferServerConfig — flashinfer is a library, no server.


# ── Adapter ──────────────────────────────────────────────────────────


_FLASHINFER_BUILD_ENV_PASSTHROUGH = ("CMAKE_BUILD_TYPE",)


class FlashinferAdapter:
    """FlashInfer framework adapter (library mode)."""

    name = "flashinfer"
    build_config_cls = FlashinferBuildConfig
    server_config_cls: type[ServerConfigBase] | None = None

    # ── Build ────────────────────────────────────────────────────────

    def build(
        self,
        repo_dir: Path,
        cfg: FlashinferBuildConfig,
        max_jobs: int,
        ctx: BuildContext,
    ) -> None:
        """Install flashinfer into repo_dir/.venv as an editable library."""
        py = str(venv_python(repo_dir))
        env: dict[str, str] = {
            "MAX_JOBS": str(max_jobs),
            "FLASHINFER_NVCC_THREADS": str(cfg.nvcc_threads),
        }
        # FlashInfer auto-detects from the GPU when this is unset.
        if cfg.cuda_arch and cfg.cuda_arch != "auto":
            env["FLASHINFER_CUDA_ARCH_LIST"] = cfg.cuda_arch

        passthrough_build_env(env, allowlist=_FLASHINFER_BUILD_ENV_PASSTHROUGH)
        setup_compiler_cache_env(env, ctx=ctx)

        # Torch first so the editable install can find it.
        install_torch_cuda(repo_dir, cfg.torch_index, ctx=ctx)

        # Submodules (3rdparty/cutlass, 3rdparty/spdlog) are required by the
        # build. clone_or_update doesn't pass --recursive, so init here.
        init_submodules(repo_dir, ctx=ctx)

        ctx.log("Building flashinfer (editable, --no-build-isolation)...")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                "--no-build-isolation",
                "-e",
                ".",
                "--extra-index-url",
                cfg.torch_index,
                "--index-strategy",
                "unsafe-best-match",
            ],
            cwd=repo_dir,
            env=env,
            ctx=ctx,
        )
        ctx.log("flashinfer build complete.")

    def build_identity(self, cfg: FlashinferBuildConfig) -> dict:
        """Cache-key fields: which knobs, when changed, force a rebuild."""
        return {
            "cuda_arch": cfg.cuda_arch,
            "nvcc_threads": cfg.nvcc_threads,
            "torch_index": cfg.torch_index,
        }

    # ── Caches ──────────────────────────────────────────────────────

    def caches_to_clear(self) -> list[Path]:
        """JIT-compiled .so files. Flashinfer builds these lazily on first use."""
        return [Path("~/.cache/flashinfer").expanduser()]

    # ── Server ──────────────────────────────────────────────────────
    # No server_spec — server_config_cls is None, so the runner never calls it.


register(FlashinferAdapter())  # type: ignore[arg-type]
