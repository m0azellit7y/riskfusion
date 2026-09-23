# RiskFusion developer commands. Requires Python 3.10+, Node 20+, PostgreSQL 14+ (see README).
PY ?= python
DATASET ?= sim-v1.1-s20260923-n5000

.PHONY: install db migrate models simulate splits register data features train evaluate score card reproduce pipeline api web build test e2e lint check

install:            ## Python + frontend dependencies
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .
	cd frontend && npm ci

db:                 ## create the local role and databases (needs a local PostgreSQL superuser)
	psql -U postgres -c "CREATE USER riskfusion WITH PASSWORD 'riskfusion' CREATEDB;" || true
	psql -U postgres -c "CREATE DATABASE riskfusion OWNER riskfusion;" || true
	psql -U postgres -c "CREATE DATABASE riskfusion_test OWNER riskfusion;" || true

migrate:            ## apply database migrations
	alembic upgrade head

models:             ## download and verify the pretrained detector models (~20 MB)
	$(PY) -m riskfusion.cli fetch-models

simulate:           ## generate the training dataset (v1.1, ~2 min, 1.2 GB) and the two evaluation sets
	$(PY) -m riskfusion.cli simulate --config configs/simulator/v1_1.yaml --skip-existing
	$(PY) -m riskfusion.cli simulate --config configs/simulator/v1_1.yaml --seed 777 --n-sessions 800 --skip-existing
	$(PY) -m riskfusion.cli simulate --config configs/simulator/v1_1_noise150.yaml --seed 777 --n-sessions 800 --skip-existing

splits:             ## create splits once (the committed manifest already exists for the default dataset)
	$(PY) -m riskfusion.cli splits $(DATASET)

register:           ## load sessions, labels and splits into PostgreSQL
	$(PY) scripts/register_dataset.py $(DATASET)

features:           ## feature store for the three datasets (~4 min)
	$(PY) -m riskfusion.cli features $(DATASET)
	$(PY) -m riskfusion.cli features sim-v1.1-s777-n800
	$(PY) -m riskfusion.cli features sim-v1.1-noise150-s777-n800

train:              ## baseline, fusion and temporal models (~1 min)
	$(PY) -m riskfusion.cli train $(DATASET)

evaluate:           ## validation evaluation, fairness, robustness, drift (~2 min)
	$(PY) -m riskfusion.cli evaluate $(DATASET)

score:              ## store assessments for the (non-holdout) simulated sessions
	$(PY) scripts/score_dataset.py $(DATASET)

card:
	$(PY) scripts/make_model_card.py

reproduce:          ## FR-39: retrain and compare with the committed model (±1 pp)
	$(PY) scripts/check_reproduction.py

# One command from raw data to final model (FR-39). The sealed holdout is NOT reopened: both permitted accesses
# are used and recorded in data/splits/.../HOLDOUT_ACCESS_LOG.jsonl.
pipeline: migrate models simulate features train evaluate register score card

data: migrate simulate register

api:                ## API on http://127.0.0.1:8000 (docs at /docs)
	uvicorn riskfusion_api.main:app --reload --port 8000

web:                ## dashboard on http://localhost:5173
	cd frontend && npm run dev

build:
	cd frontend && npm run build

test:               ## unit + integration tests (needs riskfusion_test, a trained model and `make models`)
	$(PY) -m pytest tests/unit tests/integration --cov --cov-report=term-missing:skip-covered

e2e:                ## browser test; run `make api` and `cd frontend && npm run build && npx vite preview` first
	$(PY) tests/e2e/record_flow.py --screens docs/screenshots

lint:
	ruff check src backend tests scripts && ruff format --check src backend tests scripts
	cd frontend && npx tsc -b --noEmit

check: lint test
