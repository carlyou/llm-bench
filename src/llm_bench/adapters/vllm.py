"""vLLM adapter.

Encodes how to install vLLM into a venv (precompiled-wheel or source build),
what affects the build artifact, which JIT caches it leaves behind, and how
to spell `vllm serve` for the runner's Server lifecycle.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..core.build import (
    install_torch_cuda,
    passthrough_build_env,
    setup_compiler_cache_env,
    venv_python,
)
from ..core.config import BuildConfigBase, ServerConfigBase
from ..core.executor import BuildContext, run
from ..core.server import ServerSpec
from .base import register

# ── Configs ──────────────────────────────────────────────────────────


@dataclass
class VllmBuildConfig(BuildConfigBase):
    """vllm-specific build knobs."""

    use_precompiled: bool = True
    install_flash_attn: bool = False
    install_deepgemm: bool = False
    install_flashinfer_jit_cache: bool = False
    torch_index: str = "https://download.pytorch.org/whl/cu130"


@dataclass
class VllmServerConfig(ServerConfigBase):
    """vllm serve params."""

    model: str = ""  # required
    tp: int = 1
    max_model_len: int = 4096
    enforce_eager: bool = False
    gpu_memory_utilization: float | None = None
    attention_backend: str | None = None
    compilation_config: dict | None = None
    kernel_config: dict | None = None
    log_level: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.model:
            raise ValueError("vllm server.model is required")
        if self.tp < 1:
            raise ValueError(f"vllm server.tp must be >= 1, got {self.tp}")
        if self.max_model_len < 1:
            raise ValueError(f"vllm server.max_model_len must be >= 1, got {self.max_model_len}")


# ── Adapter ──────────────────────────────────────────────────────────


_VLLM_BUILD_ENV_PASSTHROUGH = (
    "VLLM_FLASH_ATTN_SRC_DIR",
    "VLLM_CUTLASS_SRC_DIR",
    "VLLM_TARGET_DEVICE",
    "CMAKE_BUILD_TYPE",
)


class VllmAdapter:
    """vLLM framework adapter."""

    name = "vllm"
    build_config_cls = VllmBuildConfig
    server_config_cls: type[ServerConfigBase] | None = VllmServerConfig

    # ── Build ────────────────────────────────────────────────────────

    def build(
        self,
        repo_dir: Path,
        cfg: VllmBuildConfig,
        max_jobs: int,
        ctx: BuildContext,
    ) -> None:
        """Install vllm into repo_dir/.venv (precompiled wheel or source build)."""
        py = str(venv_python(repo_dir))
        uv_pip = ["uv", "pip", "install", "--python", py]
        env: dict[str, str] = {"MAX_JOBS": str(max_jobs)}

        passthrough_build_env(env, allowlist=_VLLM_BUILD_ENV_PASSTHROUGH)
        setup_compiler_cache_env(env, ctx=ctx)

        if cfg.use_precompiled:
            # Precompiled-wheel path — single-step install.
            # https://docs.vllm.ai/en/latest/contributing/#developing
            ctx.log("Installing vllm (precompiled)...")
            env["VLLM_USE_PRECOMPILED"] = "1"
            run(
                uv_pip
                + [
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
        else:
            # Source build — follows vllm contributing docs.
            ctx.log("Building vllm from source...")

            install_torch_cuda(
                repo_dir,
                cfg.torch_index,
                packages=("torch", "torchvision", "torchaudio"),
                ctx=ctx,
            )

            # Build deps from requirements/build.txt (skipping torch line).
            build_reqs = repo_dir / "requirements" / "build.txt"
            if build_reqs.exists():
                lines = []
                for raw_line in build_reqs.read_text().splitlines():
                    line = raw_line.split("#")[0].strip()
                    if line and not line.startswith("torch=="):
                        lines.append(line)
                if lines:
                    run(uv_pip + lines, ctx=ctx)

            # Optional flash-attn (build-isolated).
            if cfg.install_flash_attn:
                ctx.log("Installing flash-attn (+ build deps)...")
                run(uv_pip + ["psutil", "packaging", "ninja"], check=False, ctx=ctx)
                result = run(
                    uv_pip + ["flash-attn", "--no-build-isolation"],
                    check=False,
                    ctx=ctx,
                )
                if result.returncode != 0:
                    ctx.log("WARNING: flash-attn install failed; continuing.")

            # CUDA arch (skip when "auto" — let torch/cmake detect).
            if cfg.cuda_arch and cfg.cuda_arch != "auto":
                env["TORCH_CUDA_ARCH_LIST"] = cfg.cuda_arch
                cmake_arch = cfg.cuda_arch.replace(".", "")
                cmake_args = env.get("CMAKE_ARGS", os.environ.get("CMAKE_ARGS", ""))
                env["CMAKE_ARGS"] = f"{cmake_args} -DCMAKE_CUDA_ARCHITECTURES={cmake_arch}"

            # Build & install vllm.
            run(
                uv_pip
                + [
                    "-e",
                    ".",
                    "--no-build-isolation",
                    "--extra-index-url",
                    cfg.torch_index,
                    "--index-strategy",
                    "unsafe-best-match",
                ],
                cwd=repo_dir,
                env=env,
                ctx=ctx,
            )

        # Optional DeepGEMM.
        if cfg.install_deepgemm:
            ctx.log("Installing DeepGEMM...")
            dg_dir = repo_dir / ".deepgemm"
            if not dg_dir.exists():
                run(
                    [
                        "git",
                        "clone",
                        "--recursive",
                        "https://github.com/deepseek-ai/DeepGEMM.git",
                        str(dg_dir),
                    ],
                    ctx=ctx,
                )
            result = run(
                uv_pip + ["-e", str(dg_dir), "--no-build-isolation"],
                check=False,
                ctx=ctx,
            )
            if result.returncode != 0:
                ctx.log("WARNING: DeepGEMM install failed; continuing.")

        # Optional flashinfer-jit-cache (pre-built JIT kernels matching
        # the installed flashinfer-python version).
        if cfg.install_flashinfer_jit_cache:
            cuda_match = re.search(r"cu(\d+)", cfg.torch_index)
            cuda_ver = cuda_match.group(1) if cuda_match else "130"
            fi_index = f"https://flashinfer.ai/whl/cu{cuda_ver}"
            ver_result = run(
                [
                    py,
                    "-c",
                    "import importlib.metadata; "
                    "print(importlib.metadata.version('flashinfer-python'))",
                ],
                check=False,
                ctx=ctx,
            )
            fi_ver = ver_result.stdout.strip() if ver_result.returncode == 0 else ""
            pkg = f"flashinfer-jit-cache=={fi_ver}" if fi_ver else "flashinfer-jit-cache"
            ctx.log(f"Installing {pkg} (index: {fi_index})...")
            result = run(
                uv_pip + [pkg, "--extra-index-url", fi_index],
                check=False,
                ctx=ctx,
            )
            if result.returncode != 0:
                ctx.log("WARNING: flashinfer-jit-cache install failed; continuing.")

        ctx.log("vllm build complete.")

    def build_identity(self, cfg: VllmBuildConfig) -> dict:
        """Cache-key fields: which knobs, when changed, force a rebuild."""
        return {
            "use_precompiled": cfg.use_precompiled,
            "cuda_arch": cfg.cuda_arch,
            "install_flash_attn": cfg.install_flash_attn,
            "install_deepgemm": cfg.install_deepgemm,
            "install_flashinfer_jit_cache": cfg.install_flashinfer_jit_cache,
            "torch_index": cfg.torch_index,
        }

    # ── Caches ──────────────────────────────────────────────────────

    def caches_to_clear(self) -> list[Path]:
        """vllm + flashinfer (used internally) + triton + torchinductor."""
        home = Path("~").expanduser()
        user = os.environ.get("USER", "root")
        return [
            home / ".cache" / "vllm",
            home / ".cache" / "flashinfer",
            home / ".triton" / "cache",
            Path(f"/tmp/torchinductor_{user}"),
        ]

    # ── Server ──────────────────────────────────────────────────────

    def server_spec(
        self,
        repo_dir: Path,
        cfg: VllmServerConfig,
    ) -> ServerSpec:
        """Construct the `vllm serve <model>` command from the run's config."""
        vllm_bin = str(repo_dir / ".venv" / "bin" / "vllm")
        argv = [
            vllm_bin,
            "serve",
            cfg.model,
            "--tensor-parallel-size",
            str(cfg.tp),
            "--max-model-len",
            str(cfg.max_model_len),
            "--port",
            str(cfg.port),
            "--trust-remote-code",
        ]
        if cfg.gpu_memory_utilization is not None:
            argv += ["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)]
        if cfg.enforce_eager:
            argv.append("--enforce-eager")
        if cfg.attention_backend:
            argv += ["--attention-backend", cfg.attention_backend]
        if cfg.compilation_config:
            argv += ["-cc", json.dumps(cfg.compilation_config)]
        if cfg.kernel_config:
            argv += ["--kernel-config", json.dumps(cfg.kernel_config)]
        if cfg.log_level:
            argv += ["--log-level", cfg.log_level]

        return ServerSpec(
            argv=argv,
            env=cfg.env or {},
            cwd=repo_dir,
            health_url=f"http://127.0.0.1:{cfg.port}/v1/models",
            wait_timeout=cfg.wait_timeout,
        )


register(VllmAdapter())  # type: ignore[arg-type]
