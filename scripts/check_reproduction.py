"""FR-39: retrain from the committed configuration and check validation metrics are within ±1 percentage point
of the committed model's (docs: SRS_AUDIT A-26 — absolute percentage points)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from riskfusion.modeling.train import train_all

ROOT = Path(__file__).resolve().parents[1]
DV = "sim-v1.1-s20260923-n5000"


def main() -> int:
    ref = json.loads((ROOT / "models/trained/fusion-v1.0.0/training_results.json").read_text())
    with tempfile.TemporaryDirectory() as d:
        new = train_all(ROOT / "data", DV, Path(d) / "models", Path(d) / "runs")
    worst = 0.0
    for model in ("baseline", "primary", "temporal"):
        for m in ("pr_auc", "recall", "fpr", "ece"):
            diff = abs(new[model]["validation"][m] - ref[model]["validation"][m])
            worst = max(worst, diff)
            print(
                f"{model:9s} {m:7s} committed {ref[model]['validation'][m]:.4f} rerun {new[model]['validation'][m]:.4f}"
            )
    ok = worst <= 0.01
    print(f"largest difference {worst * 100:.2f} pp -> {'REPRODUCED' if ok else 'NOT REPRODUCED'} (tolerance 1 pp)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
