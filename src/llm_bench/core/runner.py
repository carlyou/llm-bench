"""Top-level command bodies: build, run, clean.

Framework-agnostic. Iterates over ResolvedRuns, optionally wraps each in
a Server lifecycle, executes the user's command in the venv, captures
output. Continue-on-error: a setup or command failure for one run does
not abort the whole sweep; failures are summarised at the end.

Output layout: everything for a project lives under
  <work_dir>/results/<project_name>/{build-<ts>,run-<ts>}/
A `current` symlink in the project dir points at the latest run.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from . import git
from .build import venv_dir
from .config import Config
from .executor import kill_gpu_processes
from .orchestrator import install_all
from .resolved import ResolvedRun, resolve
from .server import Server
from .summary import format_run_summary

# ── Top-level command bodies ─────────────────────────────────────────


def build_cmd(config: Config, *, timestamp: str | None = None) -> dict[str, Path]:
    """Clone and build all branches. Returns branch_name -> repo_dir.

    Build logs are written under <work_dir>/results/<project>/build-<ts>/.
    """
    ts = timestamp or _now()
    work = Path(config.project.work_dir)
    repos_dir = work / "repos" / _repo_owner_name(config.project.repo)
    build_dir = work / "results" / _safe_name(config.project.name) / f"build-{ts}"
    return install_all(config, repos_dir, logs_dir=build_dir)


def run_cmd(
    config: Config,
    *,
    config_path: Path | None = None,
    label_filter: list[str] | None = None,
    timestamp: str | None = None,
) -> dict[str, list[tuple[Path, int]]]:
    """Execute every (or filtered) run. Returns label -> [(outfile, rc), ...].

    For each run:
      1. Optionally start its server via adapter.server_spec + Server context
      2. Execute the user's command in the venv (cwd=repo_dir, PATH includes venv/bin)
      3. Capture stdout to <run_dir>/<label>[_iter{N}].txt; record exit code
      4. Tear down server, kill orphaned GPU processes

    Within a run, the iteration loop aborts on the first non-zero rc
    (subsequent iterations are skipped). Across runs, failures don't abort
    the overall sweep.

    Writes <run_dir>/summary.txt at the end.
    """
    ts = timestamp or _now()
    repos_dir = _repos_dir(config)
    branch_repo_dirs = _require_built(config, repos_dir)

    runs = resolve(config.branches, branch_repo_dirs)
    if label_filter:
        wanted = set(label_filter)
        runs = [r for r in runs if r.label in wanted]
        missing = wanted - {r.label for r in runs}
        if missing:
            raise ValueError(f"unknown run labels: {sorted(missing)}")

    run_dir = _setup_run_dir(config, ts)
    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, run_dir / "config.yaml")

    # Lazy import to avoid core ↔ adapters cycle.
    from ..adapters import get_adapter

    adapter = get_adapter(config.project.type)

    print(f"Running {len(runs)} run(s)...")
    statuses: dict[str, list[tuple[Path, int]]] = {}
    for i, rr in enumerate(runs, start=1):
        prefix = f"[run {i}/{len(runs)}] "
        statuses[rr.label] = _execute_one(rr, adapter, run_dir, prefix=prefix)

    summary = format_run_summary(config, statuses, run_dir=run_dir)
    (run_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)
    print(f"\nRun dir: {run_dir}")

    return statuses


def clean_cmd(config: Config, *, include_models: bool = False) -> None:
    """Clear framework-specific JIT caches (and optionally HF model cache)."""
    from ..adapters import get_adapter

    adapter = get_adapter(config.project.type)

    paths = list(adapter.caches_to_clear())
    if include_models:
        paths.append(Path("~/.cache/huggingface").expanduser())

    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"  cleared: {p}")
        else:
            print(f"  skipped (not a dir): {p}")


# ── Per-run execution ─────────────────────────────────────────────────


def _execute_one(
    rr: ResolvedRun,
    adapter,
    run_dir: Path,
    *,
    prefix: str,
) -> list[tuple[Path, int]]:
    """Run one ResolvedRun (with optional server). Continue-on-error: setup
    failures become a single failed iteration; per-iteration failures abort
    remaining iterations within this run, but the overall sweep continues."""
    iters = rr.run.iterations

    try:
        with _maybe_server(rr, adapter, run_dir, prefix=prefix):
            return _run_iterations(rr, run_dir, prefix=prefix, iters=iters)
    except Exception as e:
        # Setup failure (e.g. server didn't come up). Record as a synthetic failure.
        outfile = run_dir / f"{rr.label}.txt"
        outfile.write_text(f"FAILED to set up: {e}\n")
        print(f"{prefix}FAILED to set up: {e}", flush=True)
        return [(outfile, -1)]
    finally:
        kill_gpu_processes()


def _maybe_server(rr: ResolvedRun, adapter, run_dir: Path, *, prefix: str):
    """Return a Server context if the run has one configured, else nullcontext."""
    if rr.run.server is None:
        return nullcontext()
    spec = adapter.server_spec(rr.repo_dir, rr.run.server)
    log_path = run_dir / f"{rr.label}.server.log"
    return Server(spec, log_path=log_path, prefix=prefix)


def _run_iterations(
    rr: ResolvedRun,
    run_dir: Path,
    *,
    prefix: str,
    iters: int,
) -> list[tuple[Path, int]]:
    """Run the command N times. Stop early if any iteration fails."""
    out: list[tuple[Path, int]] = []
    for it in range(iters):
        suffix = f"_iter{it + 1}" if iters > 1 else ""
        it_prefix = f"{prefix.rstrip()} iter {it + 1}/{iters}] " if iters > 1 else prefix
        outfile = run_dir / f"{rr.label}{suffix}.txt"
        rc = _execute_command(rr, outfile, prefix=it_prefix)
        out.append((outfile, rc))
        if rc != 0:
            print(f"{it_prefix}aborting remaining iterations", flush=True)
            break
    return out


def _execute_command(rr: ResolvedRun, outfile: Path, *, prefix: str) -> int:
    """Run rr.run.command in the venv (PATH prepended). Return exit code."""
    venv_bin = rr.repo_dir / ".venv" / "bin"
    env = {
        **os.environ,
        "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    argv = shlex.split(rr.run.command)

    print(f"{prefix}$ {rr.run.command}", flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=rr.repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        with outfile.open("w") as out_f:
            assert proc.stdout is not None
            for line in proc.stdout:
                out_f.write(line)
                out_f.flush()
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()
            rc = proc.wait()
    finally:
        # Clean up any leaked child processes.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)

    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"{prefix}{status} → {outfile.name}", flush=True)
    return rc


# ── Helpers ───────────────────────────────────────────────────────────


def _repos_dir(config: Config) -> Path:
    return Path(config.project.work_dir) / "repos" / _repo_owner_name(config.project.repo)


def _setup_run_dir(config: Config, ts: str) -> Path:
    """Create <work_dir>/results/<project>/run-<ts>/ and update `current` symlink."""
    work = Path(config.project.work_dir)
    name = _safe_name(config.project.name)
    run_dir = work / "results" / name / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _symlink_current(run_dir)
    return run_dir


def _require_built(config: Config, repos_dir: Path) -> dict[str, Path]:
    """Verify each branch has a venv. Returns branch -> repo_dir."""
    out: dict[str, Path] = {}
    for branch_name, branch in config.branches.items():
        dir_name = git.branch_to_dir(branch_name, branch.commit)
        repo_dir = repos_dir / dir_name
        if not venv_dir(repo_dir).exists():
            raise RuntimeError(
                f"No build for branch {branch_name!r} at {repo_dir}. Run `llm-bench build` first."
            )
        out[branch_name] = repo_dir
    return out


def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_name(name: str) -> str:
    return name.replace("/", "-")


def _repo_owner_name(repo_url: str) -> str:
    m = re.search(r"[/:]([^/:]+)/([^/]+?)(?:\.git)?$", repo_url)
    return f"{m.group(1)}/{m.group(2)}" if m else "unknown/repo"


def _symlink_current(directory: Path) -> None:
    """Create '<top>/current' → relative path of `directory`.

    `top` = directory.parent.parent so the symlink lives at the project
    level, e.g. <work_dir>/results/<project>/current → run-<ts>/.
    """
    top = directory.parent.parent
    link = top / "current"
    rel = directory.relative_to(top)
    link.unlink(missing_ok=True)
    link.symlink_to(rel)
