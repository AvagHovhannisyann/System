"""SQLAlchemy declarative base shared by every ORM model."""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Deterministic constraint/index naming so Alembic autogenerate diffs stay stable."""


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base for all models.

    Every ORM model in the application must inherit from this class so a
    single ``MetaData`` (with deterministic naming) drives Alembic
    autogeneration. No models exist yet — they arrive with Phase 2.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
