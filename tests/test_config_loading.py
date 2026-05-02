"""Direct dataclass validation tests (no YAML)."""

from __future__ import annotations

import os

import pytest

from llm_bench.core.config import (
    BranchConfig,
    BuildConfigBase,
    ProjectConfig,
    RunConfig,
    ServerConfigBase,
)

# ── ProjectConfig ────────────────────────────────────────────────────


def test_project_config_requires_type():
    with pytest.raises(ValueError, match="project.type"):
        ProjectConfig(type="", repo="x")


def test_project_config_requires_repo():
    with pytest.raises(ValueError, match="project.repo"):
        ProjectConfig(type="vllm", repo="")


def test_project_config_expands_workdir(monkeypatch):
    monkeypatch.setenv("CUSTOM_HOME", "/tmp/custom")
    cfg = ProjectConfig(type="vllm", repo="x", work_dir="$CUSTOM_HOME/sub")
    assert cfg.work_dir == "/tmp/custom/sub"


def test_project_config_expands_tilde():
    cfg = ProjectConfig(type="vllm", repo="x", work_dir="~/work")
    assert cfg.work_dir == f"{os.path.expanduser('~')}/work"


# ── BuildConfigBase ──────────────────────────────────────────────────


def test_build_max_jobs_in_range():
    BuildConfigBase(max_jobs=0.5)
    BuildConfigBase(max_jobs=1.0)
    with pytest.raises(ValueError, match="max_jobs must be in"):
        BuildConfigBase(max_jobs=0.0)
    with pytest.raises(ValueError, match="max_jobs must be in"):
        BuildConfigBase(max_jobs=1.5)


def test_build_cuda_arch_required():
    with pytest.raises(ValueError, match="cuda_arch"):
        BuildConfigBase(cuda_arch="")


def test_build_default_cuda_arch_is_auto():
    assert BuildConfigBase().cuda_arch == "auto"


def test_build_default_max_jobs():
    assert BuildConfigBase().max_jobs == 0.8


# ── ServerConfigBase ─────────────────────────────────────────────────


def test_server_port_range():
    ServerConfigBase(port=8000)
    with pytest.raises(ValueError, match="server.port"):
        ServerConfigBase(port=0)
    with pytest.raises(ValueError, match="server.port"):
        ServerConfigBase(port=70000)


def test_server_wait_timeout_nonneg():
    with pytest.raises(ValueError, match="wait_timeout"):
        ServerConfigBase(wait_timeout=-1)


# ── RunConfig ────────────────────────────────────────────────────────


def test_run_label_required():
    with pytest.raises(ValueError, match="run.label"):
        RunConfig(label="", branch="main", command="x")


def test_run_command_required():
    with pytest.raises(ValueError, match="run.command"):
        RunConfig(label="smoke", branch="main", command="")


def test_run_iterations_min():
    with pytest.raises(ValueError, match="iterations"):
        RunConfig(label="smoke", branch="main", command="x", iterations=0)


# ── BranchConfig ─────────────────────────────────────────────────────


def test_branch_duplicate_labels():
    runs = [
        RunConfig(label="a", branch="b", command="x"),
        RunConfig(label="a", branch="b", command="y"),
    ]
    with pytest.raises(ValueError, match="duplicate run labels"):
        BranchConfig(name="b", build=BuildConfigBase(), runs=runs)


def test_branch_default_commit():
    branch = BranchConfig(name="main", build=BuildConfigBase())
    assert branch.commit == "HEAD"


# ── from_yaml unknown-field detection ────────────────────────────────


def test_from_yaml_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown field"):
        ProjectConfig.from_yaml(
            {"type": "vllm", "repo": "x", "bogus": True},
            source="test",
        )


def test_from_yaml_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        ProjectConfig.from_yaml(["not a dict"], source="test")
