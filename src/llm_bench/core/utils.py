"""Small framework-agnostic helpers."""

from __future__ import annotations

import multiprocessing


def resolve_jobs(max_jobs: float) -> int:
    """Resolve the (0, 1] fraction-of-cores into an absolute job count.

    Floor of (cores * fraction), at least 1. Used for the MAX_JOBS env var
    that adapters pass to ninja / make / nvcc.
    """
    return max(1, int(multiprocessing.cpu_count() * max_jobs))
