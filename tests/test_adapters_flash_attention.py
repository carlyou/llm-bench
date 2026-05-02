"""FlashAttentionAdapter unit tests."""

from __future__ import annotations

from llm_bench.adapters import get_adapter
from llm_bench.adapters.flash_attention import (
    FlashAttentionAdapter,
    FlashAttentionBuildConfig,
)


def test_registers_as_flash_attention():
    adapter = get_adapter("flash-attention")
    assert adapter.name == "flash-attention"


def test_no_server_support():
    adapter = FlashAttentionAdapter()
    assert adapter.server_config_cls is None


def test_build_identity_keys():
    adapter = FlashAttentionAdapter()
    cfg = FlashAttentionBuildConfig()
    assert set(adapter.build_identity(cfg)) == {"cuda_arch", "torch_index"}


def test_caches_to_clear_empty():
    """flash-attention is AOT-compiled; nothing JIT to clear."""
    adapter = FlashAttentionAdapter()
    assert adapter.caches_to_clear() == []


def test_build_config_yaml_parses():
    cfg = FlashAttentionBuildConfig.from_yaml(
        {"cuda_arch": "9.0a", "max_jobs": 0.8},
        source="test",
    )
    assert cfg.cuda_arch == "9.0a"
    assert cfg.max_jobs == 0.8
