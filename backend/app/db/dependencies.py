from collections.abc import Generator

from sqlalchemy.orm import Session

from ..settings import Settings
from .session import create_session_factory


def get_database_session() -> Generator[Session, None, None]:
    """Yield a request-scoped session and always close it afterward."""
    session = create_session_factory(Settings().database_url)()
    try:
        yield session
    finally:
        session.close()
