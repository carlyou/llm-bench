"""Orchestrate per-branch builds. Sequential (no parallelism per design)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO

from . import build as build_mod
from . import build_cache, git, utils
from .config import BranchConfig, Config
from .executor import BuildContext


def install_all(
    config: Config,
    repos_dir: Path,
    *,
    logs_dir: Path | None = None,
) -> dict[str, Path]:
    """Clone, set up venvs, and build each branch via the adapter.

    Skips branches where the build cache says nothing changed (matching
    commit + adapter identity). Set FORCE_BUILD=1 to bypass.

    Returns a mapping branch_name -> repo_dir. Raises on first build failure.
    """
    # Lazy import to avoid an adapters → core cycle through the adapter base.
    from ..adapters import get_adapter

    adapter = get_adapter(config.project.type)

    repos_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    n = len(config.branches)
    print(f"Installing {n} branch(es) for project {config.project.name!r}...")

    results: dict[str, Path] = {}
    for i, (branch_name, branch) in enumerate(config.branches.items(), start=1):
        prefix = f"[build {i}/{n}] " if n > 1 else ""
        with contextlib.ExitStack() as stack:
            log_fh: IO | None = None
            if logs_dir is not None:
                dir_name = git.branch_to_dir(branch_name, branch.commit)
                log_fh = stack.enter_context((logs_dir / f"build-{dir_name}.log").open("w"))
            ctx = BuildContext(prefix=prefix, log_file=log_fh)
            ctx.log("=" * 60)
            ctx.log(f"  Installing: {branch_name} @ {branch.commit}")
            ctx.log("=" * 60)
            results[branch_name] = _install_one(
                config.project.repo,
                branch_name,
                branch,
                adapter,
                repos_dir,
                ctx,
            )

    return results


def _install_one(
    repo_url: str,
    branch_name: str,
    branch: BranchConfig,
    adapter,
    repos_dir: Path,
    ctx: BuildContext,
) -> Path:
    """Clone + venv + build one branch. Honors the build-skip cache."""
    repo_dir = git.clone_or_update(
        repo_url,
        branch_name,
        branch.commit,
        repos_dir,
        auto_git_pull=branch.build.auto_git_pull,
        ctx=ctx,
    )
    build_mod.setup_venv(repo_dir, ctx=ctx)

    identity = adapter.build_identity(branch.build)
    current = build_cache.current_state(repo_dir, identity)
    if build_cache.should_skip(repo_dir, current):
        ctx.log(f"Build cached for {branch_name} (commit {current['commit'][:12]}); skipping.")
        return repo_dir

    max_jobs = utils.resolve_jobs(branch.build.max_jobs)
    adapter.build(repo_dir, branch.build, max_jobs=max_jobs, ctx=ctx)
    build_mod.check_cuda_version(repo_dir, ctx=ctx)
    build_cache.write(repo_dir, current)
    return repo_dir
