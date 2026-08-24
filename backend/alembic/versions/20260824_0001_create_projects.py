"""Create projects and project_files.

Revision ID: 20260824_0001
Revises:
"""
from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260824_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("project_number", sa.String(length=100), nullable=True),
        sa.Column("source_root", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_projects_name_nonempty"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "project_files",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("relative_path", sa.String(length=2048), nullable=False),
        sa.Column("extension", sa.String(length=50), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=10), nullable=False),
        sa.Column("asset_type", sa.String(length=17), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("parse_status", sa.String(length=7), nullable=False),
        sa.Column("parse_error_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "relative_path", name="uq_project_files_project_relative_path"),
    )
    op.create_index(op.f("ix_project_files_project_id"), "project_files", ["project_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_project_files_project_id"), table_name="project_files")
    op.drop_table("project_files")
    op.drop_table("projects")
