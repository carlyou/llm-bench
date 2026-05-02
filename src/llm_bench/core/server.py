"""Generic server lifecycle.

Adapters describe servers via ServerSpec; this module implements the
Popen + health-check + teardown logic. No framework knowledge here.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO


@dataclass
class ServerSpec:
    """Adapter-supplied description of how to start a server.

    The adapter constructs one of these from the run's typed
    `ServerConfigBase` subclass; the runner consumes it via `Server`.
    """

    argv: list[str]  # command tokens
    health_url: str  # GET this; 200 == ready
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    wait_timeout: int = 600  # seconds to wait for ready
    poll_interval: float = 1.0


class Server:
    """Generic server lifecycle (Popen + health check + teardown).

    Use as a context manager:
        with Server(spec, log_path=..., prefix="[run 1] "):
            ...  # server is up here

    On __enter__: starts the process in a new session, redirects
    stdout/stderr to log_path, polls health_url until 200 OK or timeout.
    On __exit__: SIGTERM the process group, wait, SIGKILL if needed.
    """

    def __init__(
        self,
        spec: ServerSpec,
        *,
        log_path: Path,
        prefix: str = "",
    ):
        self.spec = spec
        self.log_path = log_path
        self.prefix = prefix
        self._proc: subprocess.Popen | None = None
        self._log_fh: IO | None = None

    def __enter__(self) -> Server:
        merged_env = {**os.environ, **self.spec.env} if self.spec.env else None
        cmd_str = " ".join(self.spec.argv)
        print(f"{self.prefix}$ {cmd_str}", flush=True)

        self._log_fh = self.log_path.open("w")
        self._proc = subprocess.Popen(
            self.spec.argv,
            cwd=self.spec.cwd,
            env=merged_env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            self._wait_ready()
        except Exception:
            self._teardown()
            raise

        print(
            f"{self.prefix}Server ready (pid={self._proc.pid}); logs → {self.log_path}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._teardown()

    # ── internals ────────────────────────────────────────────────────

    def _wait_ready(self) -> None:
        deadline = time.time() + self.spec.wait_timeout
        while time.time() < deadline:
            assert self._proc is not None
            if self._proc.poll() is not None:
                rc = self._proc.returncode
                raise RuntimeError(f"server exited prematurely (rc={rc}); see {self.log_path}")
            try:
                with urllib.request.urlopen(self.spec.health_url, timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionResetError, TimeoutError):
                pass
            time.sleep(self.spec.poll_interval)
        raise TimeoutError(
            f"server didn't become ready within {self.spec.wait_timeout}s "
            f"(URL: {self.spec.health_url}); see {self.log_path}"
        )

    def _teardown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=5)
            self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
