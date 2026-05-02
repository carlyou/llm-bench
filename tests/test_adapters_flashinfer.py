"""FlashinferAdapter unit tests."""

from __future__ import annotations

import pytest

from llm_bench.adapters import get_adapter
from llm_bench.adapters.flashinfer import FlashinferAdapter, FlashinferBuildConfig
from llm_bench.core.config import Config


def test_registers_as_flashinfer():
    adapter = get_adapter("flashinfer")
    assert adapter.name == "flashinfer"


def test_no_server_support():
    adapter = FlashinferAdapter()
    assert adapter.server_config_cls is None


def test_build_identity_keys():
    adapter = FlashinferAdapter()
    cfg = FlashinferBuildConfig()
    assert set(adapter.build_identity(cfg)) == {
        "cuda_arch",
        "nvcc_threads",
        "torch_index",
    }


def test_caches_to_clear():
    adapter = FlashinferAdapter()
    paths = adapter.caches_to_clear()
    assert len(paths) == 1
    assert paths[0].name == "flashinfer"


def test_server_in_config_rejected(write_config):
    """End-to-end: setting `server:` for flashinfer should error at load time."""
    yaml = """
project: {type: flashinfer, repo: x}
build: {cuda_arch: "10.0a"}
server: {port: 8000}
branches: {main: {runs: [{label: a, command: b}]}}
"""
    p = write_config(yaml)
    with pytest.raises(ValueError, match="doesn't support a server"):
        Config.from_yaml_file(p)


def test_build_config_yaml_parses():
    cfg = FlashinferBuildConfig.from_yaml(
        {"cuda_arch": "10.0a", "nvcc_threads": 8, "max_jobs": 0.5},
        source="test",
    )
    assert cfg.cuda_arch == "10.0a"
    assert cfg.nvcc_threads == 8
    assert cfg.max_jobs == 0.5
