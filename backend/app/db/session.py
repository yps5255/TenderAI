from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


@lru_cache(maxsize=None)
def get_engine(database_url: str) -> Engine:
    """Build one engine per configured URL and enable SQLite FK enforcement."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, ":memory:"):
        Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.get_backend_name() == "sqlite" else {}
    engine = create_engine(database_url, connect_args=connect_args)

    if url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)
