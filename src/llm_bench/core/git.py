"""Git operations: clone, fetch, checkout, head SHA query."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .executor import BuildContext, run


def branch_to_dir(branch: str, commit: str = "") -> str:
    """Sanitize a branch (and optional commit prefix) for use as a dir name."""
    d = branch.replace("/", "--")
    # Skip the commit suffix when commit == "HEAD" (sentinel for branch tip).
    if commit and commit != "HEAD":
        d = f"{d}-{commit[:8]}"
    return d


def clone_or_update(
    repo_url: str,
    branch: str,
    commit: str,
    repos_dir: Path,
    *,
    auto_git_pull: bool = True,
    ctx: BuildContext | None = None,
) -> Path:
    """Clone or fetch+checkout a branch into repos_dir/<sanitized-branch>/.

    Behaviour:
      - If repo dir doesn't exist: git clone.
      - Else if auto_git_pull is False: leave alone, return path.
      - Else: ensure origin URL matches, fetch the branch.
      - If commit is "HEAD" (or empty): checkout branch + reset --hard origin/<branch>.
      - Else: checkout the specified commit/ref.

    Returns the resolved repo directory.
    """
    if ctx is None:
        ctx = BuildContext()
    dir_name = branch_to_dir(branch, commit)
    repo_dir = repos_dir / dir_name

    if not repo_dir.exists():
        ctx.log(f"Cloning {repo_url} -> {repo_dir}")
        run(["git", "clone", repo_url, str(repo_dir)], ctx=ctx)
    elif not auto_git_pull:
        ctx.log(f"auto_git_pull=false; using existing {repo_dir}")
        return repo_dir
    else:
        # Ensure origin matches (the configured repo URL might have changed).
        run(["git", "remote", "set-url", "origin", repo_url], cwd=repo_dir, ctx=ctx)

    ctx.log(f"Fetching {branch}...")
    run(["git", "fetch", "origin", branch], cwd=repo_dir, ctx=ctx)

    if not commit or commit == "HEAD":
        run(["git", "checkout", branch], cwd=repo_dir, ctx=ctx)
        run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_dir, ctx=ctx)
    else:
        run(["git", "checkout", commit], cwd=repo_dir, ctx=ctx)

    return repo_dir


def head_commit(repo_dir: Path) -> str:
    """Return the current commit SHA of repo_dir (full hash).

    Used by the build cache (component 6) for state hashing.
    """
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
    ).strip()
