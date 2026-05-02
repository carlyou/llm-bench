"""VllmAdapter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.adapters import get_adapter, known_adapters
from llm_bench.adapters.vllm import VllmAdapter, VllmBuildConfig, VllmServerConfig


def test_registers_as_vllm():
    adapter = get_adapter("vllm")
    assert adapter.name == "vllm"
    assert "vllm" in known_adapters()


def test_build_identity_keys():
    adapter = VllmAdapter()
    cfg = VllmBuildConfig()
    assert set(adapter.build_identity(cfg)) == {
        "use_precompiled",
        "cuda_arch",
        "install_flash_attn",
        "install_deepgemm",
        "install_flashinfer_jit_cache",
        "torch_index",
    }


def test_caches_to_clear():
    adapter = VllmAdapter()
    paths = adapter.caches_to_clear()
    names = {p.name for p in paths}
    assert "vllm" in names
    assert "flashinfer" in names
    # ~/.triton/cache and /tmp/torchinductor_<user>
    assert "cache" in names
    assert any(str(p).startswith("/tmp/torchinductor_") for p in paths)


def test_server_spec_argv_shape():
    adapter = VllmAdapter()
    cfg = VllmServerConfig(model="m", port=9000, tp=2, max_model_len=2048)
    spec = adapter.server_spec(Path("/tmp/repo"), cfg)
    assert spec.argv[0] == "/tmp/repo/.venv/bin/vllm"
    assert "serve" in spec.argv
    assert "m" in spec.argv
    assert "--port" in spec.argv and "9000" in spec.argv
    assert "--tensor-parallel-size" in spec.argv and "2" in spec.argv
    assert spec.health_url == "http://127.0.0.1:9000/v1/models"


def test_server_spec_optional_flags():
    adapter = VllmAdapter()
    cfg = VllmServerConfig(
        model="m",
        port=8000,
        enforce_eager=True,
        attention_backend="FLASHINFER_MLA",
        compilation_config={"pass_config": {"fuse_attn_quant": True}},
    )
    spec = adapter.server_spec(Path("/tmp/repo"), cfg)
    assert "--enforce-eager" in spec.argv
    assert "--attention-backend" in spec.argv
    assert "FLASHINFER_MLA" in spec.argv
    assert "-cc" in spec.argv


def test_server_config_requires_model():
    with pytest.raises(ValueError, match="server.model"):
        VllmServerConfig(model="")


def test_server_config_tp_min():
    with pytest.raises(ValueError, match="server.tp"):
        VllmServerConfig(model="m", tp=0)


def test_server_config_max_model_len_min():
    with pytest.raises(ValueError, match="max_model_len"):
        VllmServerConfig(model="m", max_model_len=0)
