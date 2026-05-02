# llm-bench

Pluggable A/B benchmark and test harness for LLM frameworks. One YAML config
drives clone, build, test, and benchmark across multiple branches of
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[vllm-project/flash-attention](https://github.com/vllm-project/flash-attention),
and other supported frameworks.

## Quick start

```bash
# Install dev deps
uv sync --extra dev

# Inspect a config (no side effects)
uv run llm-bench list configs/flashinfer/fp4_bench.yaml

# Build all branches in the config (auto-runs before `run`)
uv run llm-bench build configs/flashinfer/fp4_bench.yaml

# Build + execute every run, write summary.txt
uv run llm-bench run configs/flashinfer/fp4_bench.yaml

# Filter to specific runs
uv run llm-bench run configs/flashinfer/fp4_bench.yaml \
    --run main_unit --run pr1_unit

# Skip rebuild (assume builds exist)
uv run llm-bench run configs/flashinfer/fp4_bench.yaml --no-build

# Clear framework JIT caches
uv run llm-bench clean configs/flashinfer/fp4_bench.yaml
uv run llm-bench clean configs/flashinfer/fp4_bench.yaml --all  # also HF model cache
```

Force a rebuild (bypass the cache) with `FORCE_BUILD=1`.

## Config shape

Top-level structure:

```yaml
project:
  type: flashinfer        # adapter selector: "vllm" | "flashinfer" | "flash-attention"
  repo: https://github.com/...
  description: optional human-readable note

build:
  cuda_arch: "10.0a"      # or "auto"
  max_jobs: 0.8           # fraction of CPU cores

server:                   # optional; only for adapters that support it
  port: 8000
  # adapter-specific fields below — e.g. vllm: model, tp, max_model_len

branches:
  main:
    commit: HEAD          # default
    runs:
      - label: smoke
        command: pytest tests/ -v
```

Hierarchy: `build` and `server` cascade global → branch → run; later levels
override earlier ones field-by-field.

See `configs/` for full examples per adapter.

## Output layout

Every project's results live under `<work_dir>/results/<project_name>/`:

```
<work_dir>/results/<project_name>/
├── build-<ts>/
│   └── build-<branch>.log
├── run-<ts>/
│   ├── config.yaml         # snapshot of the config that produced this run
│   ├── <label>.txt         # command stdout per run
│   ├── <label>.server.log  # server stdout (if applicable)
│   └── summary.txt         # hardware info + per-run status table
└── current → run-<ts>/     # symlink to latest run
```

`work_dir` defaults to `/tmp/llm-bench`; override via `project.work_dir` in
the config (supports `~` and `$VAR` expansion).

## Development

```bash
uv run pytest               # run tests
uv run ruff check src tests # lint
uv run ruff format src tests # format
uv run mypy                 # type-check
```

## Adding a new framework adapter

1. Create `src/llm_bench/adapters/<framework>.py`
2. Subclass the relevant configs (`<X>BuildConfig`, optionally `<X>ServerConfig`)
3. Implement an adapter class (`name`, `build_config_cls`, `server_config_cls`,
   `build`, `build_identity`, `caches_to_clear`, optionally `server_spec`)
4. Call `register(<X>Adapter())` at module top-level
5. Add the new module to the lazy-import list in `adapters/base.py:get_adapter`
