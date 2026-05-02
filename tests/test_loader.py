"""YAML → Config end-to-end loader tests."""

from __future__ import annotations

import pytest

from llm_bench.core.config import Config

_FLASHINFER_MIN = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
branches:
  main:
    runs:
      - label: smoke
        command: pytest tests/
"""

_VLLM_MIN = """
project: {type: vllm, repo: x}
build: {cuda_arch: "10.0a"}
server:
  model: example/model
  port: 8000
branches:
  main:
    runs:
      - label: smoke
        command: vllm bench serve --port 8000
"""


# ── Happy paths ──────────────────────────────────────────────────────


def test_load_minimal_flashinfer(write_config):
    p = write_config(_FLASHINFER_MIN)
    cfg = Config.from_yaml_file(p)
    assert cfg.project.type == "flashinfer"
    assert cfg.project.name == f"{p.parent.name}/{p.stem}"
    assert "main" in cfg.branches
    assert cfg.branches["main"].commit == "HEAD"
    assert cfg.branches["main"].runs[0].label == "smoke"


def test_load_minimal_vllm(write_config):
    p = write_config(_VLLM_MIN)
    cfg = Config.from_yaml_file(p)
    assert cfg.project.type == "vllm"
    run = cfg.branches["main"].runs[0]
    assert run.server is not None
    assert run.server.port == 8000


def test_explicit_project_name_preserved(write_config):
    yaml = """
project:
  type: flashinfer
  repo: x
  name: my/custom/name
build: {cuda_arch: "10.0a"}
branches:
  main:
    runs: [{label: a, command: b}]
"""
    p = write_config(yaml)
    cfg = Config.from_yaml_file(p)
    assert cfg.project.name == "my/custom/name"


def test_branch_commit_pin(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
branches:
  main:
    commit: abc1234
    runs: [{label: a, command: b}]
"""
    p = write_config(yaml)
    cfg = Config.from_yaml_file(p)
    assert cfg.branches["main"].commit == "abc1234"


# ── Hierarchy merging ────────────────────────────────────────────────


def test_branch_overrides_global_build(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build:
  cuda_arch: "10.0a"
  max_jobs: 0.5
branches:
  main:
    build:
      max_jobs: 0.9
    runs: [{label: smoke, command: pytest}]
"""
    p = write_config(yaml)
    cfg = Config.from_yaml_file(p)
    assert cfg.branches["main"].build.cuda_arch == "10.0a"  # from global
    assert cfg.branches["main"].build.max_jobs == 0.9  # branch override


def test_run_overrides_branch_server(write_config):
    yaml = """
project: {type: vllm, repo: x}
build: {cuda_arch: "10.0a"}
server:
  model: m
  port: 8000
branches:
  main:
    server:
      port: 9000
    runs:
      - label: a
        command: x
      - label: b
        command: y
        server:
          port: 9100
"""
    p = write_config(yaml)
    cfg = Config.from_yaml_file(p)
    runs = cfg.branches["main"].runs
    assert runs[0].server.port == 9000  # inherits branch
    assert runs[1].server.port == 9100  # run override


# ── Error paths ──────────────────────────────────────────────────────


def test_missing_file():
    with pytest.raises(ValueError, match="not found"):
        Config.from_yaml_file("/does/not/exist.yaml")


def test_empty_file(write_config):
    p = write_config("")
    with pytest.raises(ValueError, match="empty"):
        Config.from_yaml_file(p)


def test_top_level_not_mapping(write_config):
    p = write_config("- item")
    with pytest.raises(ValueError, match="top-level must be"):
        Config.from_yaml_file(p)


def test_missing_project(write_config):
    p = write_config("branches: {main: {runs: [{label: a, command: b}]}}")
    with pytest.raises(ValueError, match="'project' is required"):
        Config.from_yaml_file(p)


def test_unknown_project_type(write_config):
    yaml = """
project: {type: unknown, repo: x}
build: {cuda_arch: "10.0a"}
branches: {main: {runs: [{label: a, command: b}]}}
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="Unknown project.type"):
        Config.from_yaml_file(p)


def test_unknown_top_level_field(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
bogus: 1
build: {cuda_arch: "10.0a"}
branches: {main: {runs: [{label: a, command: b}]}}
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="unknown top-level"):
        Config.from_yaml_file(p)


def test_unknown_branch_field(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
branches:
  main:
    bogus: 1
    runs: [{label: a, command: b}]
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="unknown field"):
        Config.from_yaml_file(p)


def test_server_on_non_server_adapter(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
server: {port: 8000}
branches: {main: {runs: [{label: a, command: b}]}}
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="doesn't support a server"):
        Config.from_yaml_file(p)


def test_empty_branches(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
branches: {}
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="non-empty"):
        Config.from_yaml_file(p)


def test_branch_no_runs(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
branches:
  main:
    runs: []
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="non-empty list"):
        Config.from_yaml_file(p)


def test_unknown_run_field(write_config):
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
branches:
  main:
    runs: [{label: a, command: b, bogus: 1}]
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="unknown"):
        Config.from_yaml_file(p)


def test_runs_property_flattens():
    """Config.runs property concatenates branch runs in order."""
    from llm_bench.adapters.flashinfer import FlashinferBuildConfig
    from llm_bench.core.config import (
        BranchConfig,
        Config,
        ProjectConfig,
        RunConfig,
    )

    proj = ProjectConfig(type="flashinfer", repo="x")
    build = FlashinferBuildConfig(cuda_arch="10.0a")
    b1 = BranchConfig(
        name="b1", build=build, runs=[RunConfig(label="r1", branch="b1", command="x")]
    )
    b2 = BranchConfig(
        name="b2", build=build, runs=[RunConfig(label="r2", branch="b2", command="y")]
    )
    cfg = Config(project=proj, branches={"b1": b1, "b2": b2})
    assert [r.label for r in cfg.runs] == ["r1", "r2"]
