"""Per-repo venv setup and shared build helpers.

Framework-agnostic build-time preparation. The actual framework install
(`pip install -e .` and friends) lives in the adapter; this module
provides the shared mechanics that adapters compose.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .executor import BuildContext, run

# ── Venv paths ───────────────────────────────────────────────────────


def venv_dir(repo_dir: Path) -> Path:
    return repo_dir / ".venv"


def venv_python(repo_dir: Path) -> Path:
    return venv_dir(repo_dir) / "bin" / "python"


# ── Venv setup ───────────────────────────────────────────────────────


def setup_venv(repo_dir: Path, *, ctx: BuildContext | None = None) -> None:
    """Create a per-repo venv with `uv venv --seed`, plus pytest.

    --seed installs pip into the venv so adapters can pip-install. pytest
    is added so run commands that invoke `pytest ...` (a common pattern)
    work without extra per-run setup. pytest is small (~5 MB).
    Idempotent — only runs on first creation.
    """
    if ctx is None:
        ctx = BuildContext()
    if not venv_dir(repo_dir).exists():
        ctx.log(f"Creating venv in {repo_dir}/.venv ...")
        run(["uv", "venv", "--python", "3.12", "--seed"], cwd=repo_dir, ctx=ctx)
        ctx.log("Installing pytest into venv...")
        run(
            ["uv", "pip", "install", "--python", str(venv_python(repo_dir)), "pytest"],
            ctx=ctx,
        )


# ── CUDA version check ───────────────────────────────────────────────


def check_cuda_version(repo_dir: Path, *, ctx: BuildContext | None = None) -> None:
    """Verify torch's CUDA version is compatible with the system's CUDA.

    Raises RuntimeError if torch CUDA is older than the system CUDA major
    (older cuBLAS may not support newer GPUs, e.g. cuBLAS 12 on SM 100+ B200).
    Logs a note when torch CUDA is newer (OK at runtime).
    No-op if either side can't be queried.
    """
    if ctx is None:
        ctx = BuildContext()

    py = str(venv_python(repo_dir))
    try:
        torch_cuda = subprocess.check_output(
            [py, "-c", "import torch; print(torch.version.cuda)"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        system_cuda = subprocess.check_output(
            ["nvcc", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return

    if torch_cuda == "None":
        raise RuntimeError(
            "CPU-only torch installed on a CUDA system. "
            "Fix: install CUDA torch, e.g.:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cu130"
        )

    m = re.search(r"release (\d+)\.", system_cuda)
    if not m:
        return
    system_major = int(m.group(1))
    torch_major = int(torch_cuda.split(".")[0])

    if torch_major < system_major:
        raise RuntimeError(
            f"CUDA version mismatch: torch has CUDA {torch_cuda} "
            f"but system has CUDA {system_major}.x. "
            f"Older cuBLAS may not support this GPU. "
            f"Fix: install torch with matching CUDA version, e.g.:\n"
            f"  pip install torch --index-url "
            f"https://download.pytorch.org/whl/cu{system_major}0"
        )
    elif torch_major > system_major:
        ctx.log(
            f"NOTE: torch has CUDA {torch_cuda}, system has "
            f"CUDA {system_major}.x (OK; torch bundles its own libs)"
        )


# ── Shared adapter helpers ──────────────────────────────────────────


def install_torch_cuda(
    repo_dir: Path,
    torch_index: str,
    *,
    packages: tuple[str, ...] = ("torch",),
    ctx: BuildContext | None = None,
) -> None:
    """Install torch (and optionally torchvision/torchaudio) into the venv
    from a CUDA-specific wheel index."""
    if ctx is None:
        ctx = BuildContext()
    py = str(venv_python(repo_dir))
    pkgs = list(packages)
    ctx.log(f"Installing {', '.join(pkgs)} from {torch_index} ...")
    run(
        ["uv", "pip", "install", "--python", py, *pkgs, "--extra-index-url", torch_index],
        ctx=ctx,
    )


def setup_compiler_cache_env(
    env: dict[str, str],
    *,
    ctx: BuildContext,
) -> None:
    """Detect sccache/ccache; if found, add to env's CMAKE_ARGS as compiler
    launcher. Mutates env in place. No-op if neither is on PATH."""
    sccache = shutil.which("sccache")
    ccache = shutil.which("ccache")
    cache_bin = sccache or ccache
    if not cache_bin:
        return
    cmake_args = env.get("CMAKE_ARGS", os.environ.get("CMAKE_ARGS", ""))
    cmake_args += (
        f" -DCMAKE_C_COMPILER_LAUNCHER={cache_bin}"
        f" -DCMAKE_CXX_COMPILER_LAUNCHER={cache_bin}"
        f" -DCMAKE_CUDA_COMPILER_LAUNCHER={cache_bin}"
    )
    env["CMAKE_ARGS"] = cmake_args
    ctx.log(f"Using {'sccache' if sccache else 'ccache'}: {cache_bin}")


def passthrough_build_env(
    env: dict[str, str],
    *,
    allowlist: tuple[str, ...],
) -> None:
    """Copy whitelisted env vars from os.environ into env if they're set.
    Mutates env in place."""
    for var in allowlist:
        if os.environ.get(var):
            env[var] = os.environ[var]


def init_submodules(repo_dir: Path, *, ctx: BuildContext | None = None) -> None:
    """Run `git submodule update --init --recursive` in repo_dir."""
    if ctx is None:
        ctx = BuildContext()
    ctx.log("Initialising submodules...")
    run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=repo_dir,
        ctx=ctx,
    )
