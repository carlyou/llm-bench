"""CLI entry for llm-bench.

Subcommands:
  build <config.yaml>           clone+build all branches in the config
  run   <config.yaml>           build (unless --no-build) then execute runs
  clean <config.yaml>           clear framework JIT caches
  list  <config.yaml>           print the run plan; don't execute anything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.config import Config
from .core.runner import build_cmd, clean_cmd, run_cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Clone and build all branches")
    p_build.add_argument("config", type=Path, help="config YAML path")

    p_run = sub.add_parser("run", help="Build (unless --no-build) then execute runs")
    p_run.add_argument("config", type=Path, help="config YAML path")
    p_run.add_argument(
        "--run",
        action="append",
        dest="labels",
        default=[],
        help="Filter to specific run labels (repeatable). If omitted, every run executes.",
    )
    p_run.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the build step (assume builds already exist).",
    )

    p_clean = sub.add_parser("clean", help="Clear framework JIT caches")
    p_clean.add_argument("config", type=Path, help="config YAML path")
    p_clean.add_argument(
        "--all",
        action="store_true",
        dest="include_models",
        help="Also clear ~/.cache/huggingface (model downloads).",
    )

    p_list = sub.add_parser("list", help="Print the run plan; don't execute")
    p_list.add_argument("config", type=Path, help="config YAML path")

    args = parser.parse_args(argv)

    try:
        config = Config.from_yaml_file(args.config)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "build":
        build_cmd(config)
        return 0

    if args.cmd == "run":
        if not args.no_build:
            build_cmd(config)
        statuses = run_cmd(
            config,
            config_path=args.config.resolve(),
            label_filter=args.labels or None,
        )
        any_failed = any(rc != 0 for iters in statuses.values() for (_, rc) in iters)
        return 1 if any_failed else 0

    if args.cmd == "clean":
        clean_cmd(config, include_models=args.include_models)
        return 0

    if args.cmd == "list":
        _print_plan(config)
        return 0

    return 0


def _print_plan(config: Config) -> None:
    """Render the branches × runs plan to stdout."""
    print(f"Project: {config.project.name}  (type={config.project.type})")
    print(f"Repo:    {config.project.repo}")
    print()
    for branch_name, branch in config.branches.items():
        print(f"Branch: {branch_name}  (commit={branch.commit})")
        for r in branch.runs:
            srv = " [server]" if r.server is not None else ""
            iter_note = f" ×{r.iterations}" if r.iterations > 1 else ""
            print(f"  - {r.label}{iter_note}{srv}")
            print(f"      {r.command}")
        print()


def _entrypoint() -> int:
    """Wrap main() to handle Ctrl-C cleanly."""
    try:
        return main()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(_entrypoint())
