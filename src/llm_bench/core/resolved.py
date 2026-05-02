"""ResolvedRun: a run paired with the per-branch build paths.

The orchestrator (or any caller that's already built the branches) returns
a list of these. Saves the runner from re-deriving repo_dir / venv paths
from raw configs every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .build import venv_python
from .config import BranchConfig, RunConfig


@dataclass
class ResolvedRun:
    """One run, plus the paths that get computed once at resolution time."""

    run: RunConfig
    branch: BranchConfig  # the branch this run belongs to
    repo_dir: Path  # where the branch is cloned

    @property
    def label(self) -> str:
        return self.run.label

    @property
    def venv_python(self) -> Path:
        return venv_python(self.repo_dir)


def resolve(
    branches: dict[str, BranchConfig],
    branch_repo_dirs: dict[str, Path],
) -> list[ResolvedRun]:
    """Flatten branches × runs into ResolvedRuns. Preserves run order.

    Raises KeyError if a branch in `branches` has no entry in
    `branch_repo_dirs` (i.e. it wasn't built).
    """
    out: list[ResolvedRun] = []
    for branch_name, branch in branches.items():
        repo_dir = branch_repo_dirs[branch_name]
        for run in branch.runs:
            out.append(ResolvedRun(run=run, branch=branch, repo_dir=repo_dir))
    return out
