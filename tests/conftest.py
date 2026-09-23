"""Shared fixtures.

Integration tests need PostgreSQL. They use RISKFUSION_TEST_DATABASE_URL (a disposable database);
the schema is rebuilt with Alembic migrations, so the tests also exercise the migrations.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = os.environ.get(
    "RISKFUSION_TEST_DATABASE_URL", "postgresql+psycopg://riskfusion:riskfusion@localhost:5432/riskfusion_test"
)


@pytest.fixture(scope="session")
def api_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    storage = tmp_path_factory.mktemp("storage")
    data = tmp_path_factory.mktemp("data")
    os.environ["RISKFUSION_DATABASE_URL"] = TEST_DB
    os.environ["RISKFUSION_STORAGE_ROOT"] = str(storage)
    os.environ["RISKFUSION_DATA_ROOT"] = str(data)
    os.environ["RISKFUSION_AUTO_ANALYSE"] = "false"  # tests trigger analysis explicitly (one test turns it on)
    from sqlalchemy import create_engine, text

    try:
        eng = create_engine(TEST_DB)
        with eng.begin() as c:
            c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
        eng.dispose()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"PostgreSQL test database not available: {e}")
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    os.environ["ALEMBIC_DATABASE_URL"] = TEST_DB
    command.upgrade(cfg, "head")
    from riskfusion_api import db as dbmod
    from riskfusion_api import deps, settings

    settings.get_settings.cache_clear()
    dbmod.get_engine.cache_clear()
    dbmod._factory.cache_clear()
    deps.get_storage.cache_clear()
    yield {"storage": str(storage), "data": str(data)}


@pytest.fixture(scope="session")
def client(api_env: dict[str, str]):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from riskfusion_api.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def tiny_webm() -> bytes:
    return (ROOT / "tests" / "fixtures" / "tiny.webm").read_bytes()
