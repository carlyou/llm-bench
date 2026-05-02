"""Hardware introspection + per-run status summary.

Framework-agnostic. Reports exit codes and result file paths; detailed
metric extraction is deferred. Raw output files are preserved in the run
directory for ad-hoc inspection.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path

from .config import Config


def hardware_info() -> dict[str, str]:
    """Best-effort system introspection. Empty strings when unavailable."""
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": _query("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"),
        "driver": _query("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "cuda": _cuda_version(),
    }


def format_run_summary(
    config: Config,
    statuses: dict[str, list[tuple[Path, int]]],
    *,
    run_dir: Path,
) -> str:
    """Render a human-readable summary string for `<run_dir>/summary.txt`."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"Project:  {config.project.name}  (type={config.project.type})")
    lines.append("")

    hw = hardware_info()
    lines.append("Hardware:")
    for k, v in hw.items():
        if v:
            lines.append(f"  {k:10s} {v}")
    lines.append("")

    lines.append("Branches:")
    for branch_name, branch in config.branches.items():
        lines.append(f"  {branch_name}  (commit={branch.commit})")
    lines.append("")

    lines.append("Status:")
    label_w = max((len(label) for label in statuses), default=0)
    for label, iters in statuses.items():
        for idx, (outfile, rc) in enumerate(iters):
            mark = "✓" if rc == 0 else "✗"
            iter_note = f" iter {idx + 1}" if len(iters) > 1 else ""
            tag = "OK" if rc == 0 else f"FAIL (rc={rc})"
            try:
                rel: Path | str = outfile.relative_to(run_dir)
            except ValueError:
                rel = outfile
            lines.append(f"  {mark} {(label + iter_note):{label_w + 8}s}  {tag:14s}  {rel}")

    failures = sum(1 for iters in statuses.values() for (_, rc) in iters if rc != 0)
    total = sum(len(iters) for iters in statuses.values())
    lines.append("")
    lines.append(f"Summary: {total - failures}/{total} OK")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Internals ────────────────────────────────────────────────────────


def _query(*cmd: str) -> str:
    """Run a command; return stripped stdout, or empty on failure."""
    try:
        return subprocess.check_output(
            list(cmd),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _cuda_version() -> str:
    """Extract CUDA release version from `nvcc --version`."""
    out = _query("nvcc", "--version")
    if not out:
        return ""
    m = re.search(r"release ([\d.]+)", out)
    return m.group(1) if m else ""
