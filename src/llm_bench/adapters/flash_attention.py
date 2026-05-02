"""vLLM-flavored flash-attention adapter.

Encodes how to install vllm-project/flash-attention into a venv. It's a
torch C++ extension, AOT-compiled at install time via setuptools+ninja.
No server; library mode only.
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
class FlashAttentionBuildConfig(BuildConfigBase):
    """flash-attention build knobs."""

    torch_index: str = "https://download.pytorch.org/whl/cu130"


# No FlashAttentionServerConfig — flash-attention is a library, no server.


# ── Adapter ──────────────────────────────────────────────────────────


_FA_BUILD_ENV_PASSTHROUGH = (
    "CMAKE_BUILD_TYPE",
    "FLASH_ATTENTION_FORCE_CXX11_ABI",
    "FLASH_ATTENTION_FORCE_BUILD",
)


class FlashAttentionAdapter:
    """vllm-project/flash-attention adapter (library mode)."""

    name = "flash-attention"
    build_config_cls = FlashAttentionBuildConfig
    server_config_cls: type[ServerConfigBase] | None = None

    # ── Build ────────────────────────────────────────────────────────

    def build(
        self,
        repo_dir: Path,
        cfg: FlashAttentionBuildConfig,
        max_jobs: int,
        ctx: BuildContext,
    ) -> None:
        """Install flash-attention into repo_dir/.venv as an editable extension."""
        py = str(venv_python(repo_dir))
        env: dict[str, str] = {"MAX_JOBS": str(max_jobs)}
        # torch's setuptools extension uses TORCH_CUDA_ARCH_LIST.
        if cfg.cuda_arch and cfg.cuda_arch != "auto":
            env["TORCH_CUDA_ARCH_LIST"] = cfg.cuda_arch

        passthrough_build_env(env, allowlist=_FA_BUILD_ENV_PASSTHROUGH)
        setup_compiler_cache_env(env, ctx=ctx)

        # Torch first so the extension build can import it during setup.
        install_torch_cuda(repo_dir, cfg.torch_index, ctx=ctx)

        # cutlass submodule is required by the kernels.
        init_submodules(repo_dir, ctx=ctx)

        ctx.log("Building flash-attention (editable, --no-build-isolation)...")
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
        ctx.log("flash-attention build complete.")

    def build_identity(self, cfg: FlashAttentionBuildConfig) -> dict:
        """Cache-key fields: which knobs, when changed, force a rebuild."""
        return {
            "cuda_arch": cfg.cuda_arch,
            "torch_index": cfg.torch_index,
        }

    # ── Caches ──────────────────────────────────────────────────────

    def caches_to_clear(self) -> list[Path]:
        """flash-attention is AOT-compiled at install time; no JIT cache."""
        return []

    # ── Server ──────────────────────────────────────────────────────
    # No server_spec — server_config_cls is None.


register(FlashAttentionAdapter())  # type: ignore[arg-type]
