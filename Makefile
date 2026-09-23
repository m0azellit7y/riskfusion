# RiskFusion developer commands. Requires Python 3.10+, Node 20+, PostgreSQL 14+ (see README).
PY ?= python
DATASET ?= sim-v1-s20260923-n5000

.PHONY: install db migrate simulate splits register data api web build test e2e lint check

install:            ## Python + frontend dependencies
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .
	cd frontend && npm ci

db:                 ## create the local role and databases (needs a local PostgreSQL superuser)
	psql -U postgres -c "CREATE USER riskfusion WITH PASSWORD 'riskfusion' CREATEDB;" || true
	psql -U postgres -c "CREATE DATABASE riskfusion OWNER riskfusion;" || true
	psql -U postgres -c "CREATE DATABASE riskfusion_test OWNER riskfusion;" || true

migrate:            ## apply database migrations
	alembic upgrade head

simulate:           ## generate the 5,000-session dataset (about 2 minutes, 1.2 GB)
	$(PY) -m riskfusion.cli simulate

splits:             ## create splits once (the committed manifest already exists for the default dataset)
	$(PY) -m riskfusion.cli splits $(DATASET)

register:           ## load sessions, labels and splits into PostgreSQL
	$(PY) scripts/register_dataset.py $(DATASET)

data: migrate simulate register

api:                ## API on http://127.0.0.1:8000 (docs at /docs)
	uvicorn riskfusion_api.main:app --reload --port 8000

web:                ## dashboard on http://localhost:5173
	cd frontend && npm run dev

build:
	cd frontend && npm run build

test:               ## unit + integration tests (integration needs riskfusion_test)
	$(PY) -m pytest tests/unit tests/integration --cov --cov-report=term-missing:skip-covered

e2e:                ## browser test; run `make api` and `cd frontend && npm run build && npx vite preview` first
	$(PY) tests/e2e/record_flow.py --screens docs/screenshots

lint:
	ruff check src backend tests scripts && ruff format --check src backend tests scripts
	cd frontend && npx tsc -b --noEmit

check: lint test
