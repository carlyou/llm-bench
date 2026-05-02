"""Subprocess runner with line-by-line streaming and log capture.

Also houses GPU process cleanup (used between runs).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass
class BuildContext:
    """Logging context passed through build/run operations.

    `prefix` is prepended to every printed line. `log_file`, if set,
    receives the same lines without the prefix.
    """

    prefix: str = ""
    log_file: IO | None = None

    def log(self, msg: str) -> None:
        for line in msg.splitlines():
            print(f"{self.prefix}{line}", flush=True)
            if self.log_file:
                self.log_file.write(f"{line}\n")


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    ctx: BuildContext | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, streaming each output line in real time.

    `env` (when provided) is merged on top of os.environ. Stderr is folded
    into stdout so log ordering is preserved.
    """
    if ctx is None:
        ctx = BuildContext()
    merged_env = {**os.environ, **(env or {})} if env else None

    cmd_str = " ".join(cmd)
    print(f"{ctx.prefix}$ {cmd_str}", flush=True)
    if ctx.log_file:
        ctx.log_file.write(f"$ {cmd_str}\n")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        output_lines.append(line)
        print(f"{ctx.prefix}{line}", flush=True)
        if ctx.log_file:
            ctx.log_file.write(f"{line}\n")
    rc = proc.wait()

    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    return subprocess.CompletedProcess(
        cmd,
        rc,
        stdout="\n".join(output_lines),
        stderr=None,
    )


def kill_gpu_processes(timeout_s: int = 30) -> None:
    """Kill any leftover GPU compute processes; wait for memory release.

    Used between runs so the next run starts on a clean GPU.
    No-op if nvidia-smi isn't available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError, ValueError):
                os.kill(int(pid), signal.SIGKILL)
        if pids:
            for _ in range(timeout_s):
                time.sleep(1)
                probe = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if not probe.stdout.strip():
                    return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
