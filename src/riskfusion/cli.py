"""Command-line entry point: ``riskfusion <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _data_root(arg: str | None) -> Path:
    return Path(arg or os.environ.get("RISKFUSION_DATA_ROOT", "data")).resolve()


def cmd_simulate(a: argparse.Namespace) -> int:
    from riskfusion.simulator import generate, load_config

    cfg = load_config(a.config, n_sessions=a.n_sessions, seed=a.seed)
    res = generate(cfg, _data_root(a.data_root), overwrite=a.overwrite)
    print(json.dumps(res.manifest, indent=2))
    return 0


def cmd_splits(a: argparse.Namespace) -> int:
    from riskfusion.data.splits import create_splits_for_dataset

    manifest = create_splits_for_dataset(_data_root(a.data_root), a.dataset_version, seed=a.seed)
    print(json.dumps({k: v for k, v in manifest.items() if k != "holdout_session_ids"}, indent=2))
    return 0


def cmd_schemas(a: argparse.Namespace) -> int:
    from riskfusion.contracts import write_schemas

    write_schemas(Path(a.out))
    print(f"schemas written to {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="riskfusion")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a simulated dataset version (FR-1)")
    s.add_argument("--config", default="configs/simulator/v1.yaml")
    s.add_argument("--n-sessions", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--data-root", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("splits", help="create train/validation/calibration/sealed-holdout splits (DR-5)")
    s.add_argument("dataset_version")
    s.add_argument("--seed", type=int, default=20260923)
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_splits)

    s = sub.add_parser("schemas", help="regenerate contract JSON Schemas")
    s.add_argument("--out", default="contracts")
    s.set_defaults(func=cmd_schemas)

    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
