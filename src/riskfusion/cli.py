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
    from riskfusion.simulator.runner import run_id_for

    cfg = load_config(a.config, n_sessions=a.n_sessions, seed=a.seed)
    if a.skip_existing and (_data_root(a.data_root) / "synthetic" / run_id_for(cfg) / "manifest.json").exists():
        print(f"{run_id_for(cfg)} exists; skipped")
        return 0
    res = generate(cfg, _data_root(a.data_root), overwrite=a.overwrite)
    print(json.dumps(res.manifest, indent=2))
    return 0


def cmd_splits(a: argparse.Namespace) -> int:
    from riskfusion.data.splits import create_splits_for_dataset

    manifest = create_splits_for_dataset(_data_root(a.data_root), a.dataset_version, seed=a.seed)
    print(json.dumps({k: v for k, v in manifest.items() if k != "holdout_session_ids"}, indent=2))
    return 0


def cmd_features(a: argparse.Namespace) -> int:
    from riskfusion.features.build import build_feature_store

    s = build_feature_store(_data_root(a.data_root), a.dataset_version)
    print(json.dumps({k: v for k, v in s.items() if k != "columns"}, indent=2))
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from riskfusion.modeling.train import train_all

    r = train_all(_data_root(a.data_root), a.dataset_version, Path(a.models), Path(a.runs))
    print(json.dumps({k: r[k]["validation"] for k in ("baseline", "primary", "temporal")}, indent=2, default=float))
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from riskfusion.modeling.evaluate import evaluate_all
    from riskfusion.modeling.monitoring import run_monitoring

    root = _data_root(a.data_root)
    evaluate_all(root, a.dataset_version, Path(a.model), Path(a.out))
    run_monitoring(root, Path(a.model), Path(a.out))
    print(f"evaluation written to {a.out}")
    return 0


def cmd_holdout(a: argparse.Namespace) -> int:
    from riskfusion.modeling.evaluate import holdout_evaluation

    r = holdout_evaluation(_data_root(a.data_root), a.dataset_version, Path(a.model), Path(a.out), a.access, a.purpose)
    print(json.dumps({k: r[k] for k in ("n", "positives", "primary", "baseline")}, indent=2, default=float))
    return 0


def cmd_score_dir(a: argparse.Namespace) -> int:
    from riskfusion.serving import score_directory

    print(json.dumps(score_directory(Path(a.input), Path(a.output), Path(a.model)), indent=2))
    return 0


def cmd_fetch_models(a: argparse.Namespace) -> int:
    from riskfusion.extract.models import fetch_models

    print(json.dumps(fetch_models(), indent=2))
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
    s.add_argument("--skip-existing", action="store_true", help="do nothing if this dataset version exists")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("splits", help="create train/validation/calibration/sealed-holdout splits (DR-5)")
    s.add_argument("dataset_version")
    s.add_argument("--seed", type=int, default=20260923)
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_splits)

    dv11 = "sim-v1.1-s20260923-n5000"
    model = "models/trained/fusion-v1.0.0"
    s = sub.add_parser("features", help="build the versioned feature store (FR-15)")
    s.add_argument("dataset_version", nargs="?", default=dv11)
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_features)
    s = sub.add_parser("train", help="train baseline, primary and temporal models (Phase 4)")
    s.add_argument("dataset_version", nargs="?", default=dv11)
    s.add_argument("--models", default="models/trained")
    s.add_argument("--runs", default="runs")
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_train)
    s = sub.add_parser("evaluate", help="validation evaluation, fairness, robustness, drift (Phases 5-6)")
    s.add_argument("dataset_version", nargs="?", default=dv11)
    s.add_argument("--model", default=model)
    s.add_argument("--out", default="reports/evaluation")
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_evaluate)
    s = sub.add_parser("holdout", help="open the sealed holdout (logged; at most twice)")
    s.add_argument("--access", type=int, choices=(1, 2), required=True)
    s.add_argument("--purpose", required=True)
    s.add_argument("--dataset-version", default=dv11)
    s.add_argument("--model", default=model)
    s.add_argument("--out", default="reports/evaluation")
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_holdout)
    s = sub.add_parser("score-dir", help="batch-score a directory of sessions (FR-36)")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--model", default=model)
    s.set_defaults(func=cmd_score_dir)
    s = sub.add_parser("fetch-models", help="download and verify the pretrained detector models")
    s.set_defaults(func=cmd_fetch_models)

    s = sub.add_parser("schemas", help="regenerate contract JSON Schemas")
    s.add_argument("--out", default="contracts")
    s.set_defaults(func=cmd_schemas)

    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
