from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from riskfusion_api.models import Base
from riskfusion_api.settings import get_settings

config = context.config
url = os.environ.get("ALEMBIC_DATABASE_URL") or get_settings().database_url
config.set_main_option("sqlalchemy.url", url)
target_metadata = Base.metadata


def run_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
