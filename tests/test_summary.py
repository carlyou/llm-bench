"""Summary formatter unit tests."""

from __future__ import annotations

from pathlib import Path

from llm_bench.adapters.flashinfer import FlashinferBuildConfig
from llm_bench.core.config import (
    BranchConfig,
    Config,
    ProjectConfig,
    RunConfig,
)
from llm_bench.core.summary import format_run_summary, hardware_info


def _make_config():
    proj = ProjectConfig(type="flashinfer", repo="https://example.com/x.git")
    proj.name = "test/example"
    build = FlashinferBuildConfig(cuda_arch="10.0a")
    runs = [RunConfig(label="unit", branch="main", command="pytest tests/")]
    branch = BranchConfig(name="main", build=build, commit="HEAD", runs=runs)
    return Config(project=proj, branches={"main": branch})


def test_hardware_info_keys():
    info = hardware_info()
    assert set(info) >= {"platform", "python", "gpu", "driver", "cuda"}


def test_format_run_summary_ok():
    cfg = _make_config()
    statuses = {"unit": [(Path("/tmp/run/unit.txt"), 0)]}
    out = format_run_summary(cfg, statuses, run_dir=Path("/tmp/run"))
    assert "test/example" in out
    assert "✓" in out
    assert "OK" in out
    assert "unit.txt" in out
    assert "1/1 OK" in out


def test_format_run_summary_failure():
    cfg = _make_config()
    statuses = {"unit": [(Path("/tmp/run/unit.txt"), 1)]}
    out = format_run_summary(cfg, statuses, run_dir=Path("/tmp/run"))
    assert "✗" in out
    assert "FAIL (rc=1)" in out
    assert "0/1 OK" in out


def test_format_run_summary_iterations():
    cfg = _make_config()
    statuses = {
        "unit": [
            (Path("/tmp/run/unit_iter1.txt"), 0),
            (Path("/tmp/run/unit_iter2.txt"), 0),
            (Path("/tmp/run/unit_iter3.txt"), 1),
        ]
    }
    out = format_run_summary(cfg, statuses, run_dir=Path("/tmp/run"))
    assert "iter 1" in out
    assert "iter 3" in out
    assert "2/3 OK" in out


def test_format_run_summary_empty():
    cfg = _make_config()
    out = format_run_summary(cfg, {}, run_dir=Path("/tmp/run"))
    assert "0/0 OK" in out
