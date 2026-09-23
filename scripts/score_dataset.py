"""Store risk assessments for every non-holdout session of a registered simulated dataset.

Usage: python scripts/score_dataset.py sim-v1.1-s20260923-n5000
"""

from __future__ import annotations

import json
import sys

from riskfusion_api.db import _factory
from riskfusion_api.services.analysis import score_simulated
from riskfusion_api.settings import get_settings

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with _factory()() as db:
        print(json.dumps(score_simulated(db, get_settings().data_root.resolve(), sys.argv[1]), indent=2))
