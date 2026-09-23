"""Register a generated simulated dataset (and its splits) into PostgreSQL.

Usage: python scripts/register_dataset.py sim-v1-s20260923-n5000
"""

from __future__ import annotations

import json
import sys

from riskfusion_api.db import _factory
from riskfusion_api.services.datasets import register_simulated
from riskfusion_api.settings import get_settings

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with _factory()() as db:
        print(json.dumps(register_simulated(db, get_settings().data_root.resolve(), sys.argv[1]), indent=2))
