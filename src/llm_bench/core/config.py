"""Config dataclasses with YAML-aware factory classmethods.

Loading a config from a YAML file goes through `Config.from_yaml_file(path)`.
Each dataclass exposes a `from_yaml(raw, *, source=...)` classmethod that
parses a (sub)mapping into a typed instance. Simple configs inherit the
default implementation from `ConfigBase`; orchestrating configs (Config,
BranchConfig, RunConfig) override to coordinate child sections and adapter
dispatch.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import yaml

# ── Shared base ──────────────────────────────────────────────────────


class ConfigBase:
    """Base for all config dataclasses.

    Provides a default `from_yaml` that field-filters a mapping and
    constructs the dataclass. Subclasses override when they need to
    orchestrate child sections (BranchConfig, Config, RunConfig).
    """

    @classmethod
    def from_yaml(cls, raw: dict | None, *, source: str = "") -> Self:
        if raw is None:
            raise ValueError(f"{source}: section is required (got null)")
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: must be a mapping, got {type(raw).__name__}")
        valid = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(
                f"{source}: unknown field(s) {sorted(unknown)} for "
                f"{cls.__name__}. Valid: {sorted(valid)}"
            )
        return cls(**{k: v for k, v in raw.items() if k in valid})


# ── Generic base configs ─────────────────────────────────────────────


@dataclass
class ProjectConfig(ConfigBase):
    """Top-level project metadata. Drives adapter dispatch."""

    type: str  # required: "vllm" | "flashinfer" | ...
    repo: str  # required: git URL
    work_dir: str = "/tmp/llm-bench"
    name: str = ""  # derived from config path if empty
    description: str = ""

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("project.type is required")
        if not self.repo:
            raise ValueError("project.repo is required")
        # Expand ~ and env vars so configs can use ~/work or $HOME/...
        self.work_dir = os.path.expandvars(os.path.expanduser(self.work_dir))


@dataclass
class BuildConfigBase(ConfigBase):
    """Fields common to all framework build configs."""

    cuda_arch: str = "auto"  # "auto" -> adapter resolves from GPU
    max_jobs: float = 0.8  # fraction of CPU cores; (0, 1]
    auto_git_pull: bool = True

    def __post_init__(self) -> None:
        if not self.cuda_arch:
            raise ValueError("build.cuda_arch is required (use 'auto' to auto-detect)")
        if not (0 < self.max_jobs <= 1):
            raise ValueError(f"build.max_jobs must be in (0, 1], got {self.max_jobs}")


@dataclass
class ServerConfigBase(ConfigBase):
    """Generic server params. Adapter subclasses add framework-specific fields."""

    port: int = 8000
    wait_timeout: int = 600  # seconds to wait for health
    env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"server.port must be in [1, 65535], got {self.port}")
        if self.wait_timeout < 0:
            raise ValueError(f"server.wait_timeout must be >= 0, got {self.wait_timeout}")


# ── Run / Branch / Config ────────────────────────────────────────────


@dataclass
class RunConfig(ConfigBase):
    """One execution unit. Runs the command in the per-branch venv."""

    label: str  # required, unique within branch
    branch: str  # parent branch name (filled by from_yaml)
    command: str = ""  # required: shell command to execute
    iterations: int = 1  # repeat the command N times
    server: ServerConfigBase | None = None  # if set, start server before command

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("run.label is required")
        if not self.command:
            raise ValueError(f"run.command is required (label={self.label!r})")
        if self.iterations < 1:
            raise ValueError(f"run.iterations must be >= 1, got {self.iterations}")

    @classmethod
    def from_yaml(  # type: ignore[override]
        cls,
        raw: dict,
        *,
        branch_name: str,
        run_index: int,
        adapter,
        branch_server: dict | None,
        source: str = "",
    ) -> RunConfig:
        where = f"{source}.runs[{run_index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a mapping")
        label = raw.get("label", f"<runs[{run_index}]>")

        # Server: merge branch+global with run-level overrides, so a run can
        # change individual fields (e.g. just the port) without restating
        # everything.
        run_server_raw = raw.get("server")
        if run_server_raw is not None and not isinstance(run_server_raw, dict):
            raise ValueError(f"{where} ({label!r}).server: must be a mapping")
        server_dict: dict | None
        if run_server_raw is not None:
            server_dict = {**(branch_server or {}), **run_server_raw}
        else:
            server_dict = branch_server

        server_obj: ServerConfigBase | None = None
        if server_dict is not None:
            if adapter.server_config_cls is None:
                raise ValueError(
                    f"{where} ({label!r}).server: adapter {adapter.name!r} doesn't support a server"
                )
            server_obj = adapter.server_config_cls.from_yaml(
                server_dict, source=f"{where} ({label!r}).server"
            )

        # Field-filter the run's own keys (excluding 'server', set above).
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(
                f"{where} ({label!r}): unknown field(s) {sorted(unknown)} "
                f"for {cls.__name__}. Valid: {sorted(valid)}"
            )
        kwargs = {k: v for k, v in raw.items() if k in valid and k != "server"}
        kwargs["branch"] = branch_name
        kwargs["server"] = server_obj
        return cls(**kwargs)


_BRANCH_YAML_KEYS = {"commit", "build", "server", "runs"}


@dataclass
class BranchConfig(ConfigBase):
    """Per-branch overrides + the run list for that branch."""

    name: str
    build: BuildConfigBase
    commit: str = "HEAD"  # "HEAD" = branch tip; else SHA/ref
    runs: list[RunConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or not self.commit:
            raise ValueError(f"branch {self.name!r}: commit must be a non-empty string")
        labels = [r.label for r in self.runs]
        if len(labels) != len(set(labels)):
            duplicates = sorted({lbl for lbl in labels if labels.count(lbl) > 1})
            raise ValueError(f"branch {self.name!r}: duplicate run labels: {duplicates}")

    @classmethod
    def from_yaml(  # type: ignore[override]
        cls,
        name: str,
        raw: dict,
        *,
        adapter,
        global_build: dict,
        global_server: dict | None,
        source: str = "",
    ) -> BranchConfig:
        where = f"{source}: branches.{name}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a mapping")

        unknown = set(raw) - _BRANCH_YAML_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown field(s) {sorted(unknown)}. Valid: {sorted(_BRANCH_YAML_KEYS)}"
            )

        # Commit (defaults to "HEAD" = branch tip).
        commit = raw.get("commit", "HEAD")
        if not isinstance(commit, str):
            raise ValueError(f"{where}.commit: must be a string, got {type(commit).__name__}")

        # Build: global → branch.
        branch_build = raw.get("build") or {}
        if not isinstance(branch_build, dict):
            raise ValueError(f"{where}.build: must be a mapping")
        merged_build = {**global_build, **branch_build}
        build_cfg = adapter.build_config_cls.from_yaml(merged_build, source=f"{where}.build")

        # Server: global → branch (run can override).
        branch_server = raw.get("server")
        if branch_server is not None and not isinstance(branch_server, dict):
            raise ValueError(f"{where}.server: must be a mapping")
        if branch_server is not None and adapter.server_config_cls is None:
            raise ValueError(f"{where}.server: adapter {adapter.name!r} doesn't support a server")
        effective_server = (
            {**(global_server or {}), **branch_server}
            if branch_server is not None
            else global_server
        )

        # Runs.
        runs_raw = raw.get("runs")
        if not isinstance(runs_raw, list) or not runs_raw:
            raise ValueError(
                f"{where}.runs: must be a non-empty list (branch with no runs has nothing to do)"
            )
        runs = [
            RunConfig.from_yaml(
                r,
                branch_name=name,
                run_index=i,
                adapter=adapter,
                branch_server=effective_server,
                source=where,
            )
            for i, r in enumerate(runs_raw)
        ]
        return cls(name=name, build=build_cfg, commit=commit, runs=runs)


_CONFIG_YAML_KEYS = {"project", "build", "server", "branches"}


@dataclass
class Config(ConfigBase):
    """Top-level resolved config object."""

    project: ProjectConfig
    branches: dict[str, BranchConfig]

    @property
    def runs(self) -> list[RunConfig]:
        """Flat list of all runs across all branches."""
        return [r for b in self.branches.values() for r in b.runs]

    @classmethod
    def from_yaml_file(cls, path: Path | str) -> Config:
        """Load and validate a config from a YAML file."""
        path = Path(path).resolve()
        if not path.exists():
            raise ValueError(f"config file not found: {path}")
        with path.open() as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"config file is empty: {path}")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level must be a YAML mapping, got {type(raw).__name__}")
        default_name = f"{path.parent.name}/{path.stem}"
        return cls.from_yaml(raw, default_name=default_name, source=str(path))

    @classmethod
    def from_yaml(  # type: ignore[override]
        cls,
        raw: dict,
        *,
        default_name: str = "",
        source: str = "",
    ) -> Config:
        # Lazy import: adapters/base.py imports BuildConfigBase /
        # ServerConfigBase from this module, so a top-level import here
        # would create a cycle.
        from ..adapters import get_adapter

        unknown = set(raw) - _CONFIG_YAML_KEYS
        if unknown:
            raise ValueError(
                f"{source}: unknown top-level field(s) {sorted(unknown)}. "
                f"Valid: {sorted(_CONFIG_YAML_KEYS)}"
            )

        # 1. Project (selects the adapter).
        project_raw = raw.get("project")
        if project_raw is None:
            raise ValueError(f"{source}: top-level 'project' is required")
        project = ProjectConfig.from_yaml(project_raw, source=f"{source}: project")
        if not project.name:
            project.name = default_name

        adapter = get_adapter(project.type)

        # 2. Globals (defaults applied to every branch).
        global_build = raw.get("build") or {}
        if not isinstance(global_build, dict):
            raise ValueError(f"{source}: 'build' must be a mapping")
        global_server = raw.get("server")
        if global_server is not None and not isinstance(global_server, dict):
            raise ValueError(f"{source}: 'server' must be a mapping")
        if global_server is not None and adapter.server_config_cls is None:
            raise ValueError(
                f"{source}: 'server' is set but adapter {adapter.name!r} doesn't support a server"
            )

        # 3. Branches.
        branches_raw = raw.get("branches")
        if not isinstance(branches_raw, dict) or not branches_raw:
            raise ValueError(f"{source}: 'branches' must be a non-empty mapping")
        branches = {
            name: BranchConfig.from_yaml(
                name,
                branch_raw or {},
                adapter=adapter,
                global_build=global_build,
                global_server=global_server,
                source=source,
            )
            for name, branch_raw in branches_raw.items()
        }

        return cls(project=project, branches=branches)
