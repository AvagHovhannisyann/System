"""Extraction framework: prompt versions, activations, golden scores, results (P7.3/P7.10).

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-01

Four tables, none bitemporal, none a hypertable, all **append-only** by the
same ``BEFORE UPDATE OR DELETE`` row trigger shape revisions 0003/0004/0007/0009
use:

- ``extraction_prompt_version`` — one row per ``(name, version_hash)``. A
  prompt version is content-addressed (§6.5, P7.10), so its identity *is* its
  content: saving is idempotent on the address and editing a prompt produces a
  different row rather than changing this one. Nothing is updated, ever, which
  is what lets a golden-set score stay attached to the text it actually
  measured.
- ``extraction_prompt_activation`` — one row per act of pointing a prompt at a
  version. "Which version is in force" is a pointer, and moving it is a
  configuration change, so it is an event (§6.11: config changes are events,
  not mutations). §6.5's one-click **rollback is an activation naming an
  earlier hash** — no restore, no rewrite, so there is nothing to lose.
- ``extraction_golden_score`` — one row per golden-set run against one version,
  foreign-keyed to the version so a score can never come to describe different
  text. Deliberately **no threshold and no pass/fail column**: per D-014 the
  directive's "≥ 85%" is the operator's prior and the human noise floor is
  unmeasured (B3), so a stored bar would be a fabricated number where a reader
  would take it as authoritative (I3).
- ``extraction_result`` — one row per model call, holding the raw response
  verbatim (§5-P7) plus the prompt version hash and model that produced it.
  Re-running is a new row, never an overwrite: two rows disagreeing about one
  address falsifies the determinism the cache design rests on, and that is
  exactly the observation worth keeping.

None is bitemporal, and that is deliberate. The bitemporal columns describe
when a fact was true in the world and when it became knowable to the *market*
(D-011). A prompt version, an activation, a score and an extraction result are
all things **we** did to our own system: they have no market knowability, and a
``knowledge_time`` invented for them would be a fabricated value in the one
column whose meaning is that it is not fabricated (I3). Same reasoning as
revisions 0005, 0007 and 0009, so they are absent from the bitemporal registry,
unscoped by the Core-level read guard, and read without an as-of.

None is a hypertable either: a prompt library is a human-sized collection, and
extraction results are addressed by document rather than along an event-time
axis worth chunking on.

TRUNCATE stays unblocked on all four, matching 0003/0004/0007/0009: it is the
sanctioned admin/test reset path and never masquerades as an edit.

Downgrade drops the triggers, their function, and all four tables in dependency
order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = (
    "extraction_prompt_version",
    "extraction_prompt_activation",
    "extraction_golden_score",
    "extraction_result",
)
"""Tables that get the append-only trigger, in creation order.

Named here *and* spelled out in full ``CREATE TRIGGER`` statements below rather
than generated from this tuple. The duplication is deliberate: "which tables in
this schema are append-only" is a question a human answers by grepping for
``BEFORE UPDATE OR DELETE ON``, and a loop that assembles the statement makes
that grep return nothing. Revisions 0003, 0004, 0007 and 0009 write theirs out
for the same reason, and a test asserts the statements are present.
"""


def upgrade() -> None:
    """Create the four extraction tables, their constraints, and the append-only triggers."""
    op.create_table(
        "extraction_prompt_version",
        sa.Column("prompt_version_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version_hash", sa.Text(), nullable=False),
        sa.Column("system", sa.Text(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("schema_digest", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("prompt_version_id", name="pk_extraction_prompt_version"),
        # The content address is the logical key, so it is unique per prompt.
        # This constraint is also the FK target for activations and scores, and
        # the lookup path for "give me this exact version" — no separate index.
        sa.UniqueConstraint("name", "version_hash", name="uq_extraction_prompt_version_address"),
        sa.UniqueConstraint("name", "sequence", name="uq_extraction_prompt_version_sequence"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005, 0007 and 0009).
        sa.CheckConstraint("name <> ''", name="name_not_empty"),
        sa.CheckConstraint("version_hash <> ''", name="version_hash_not_empty"),
        sa.CheckConstraint("template <> ''", name="template_not_empty"),
        sa.CheckConstraint("schema_digest <> ''", name="schema_digest_not_empty"),
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        sa.CheckConstraint("sequence >= 1", name="sequence_positive"),
    )
    op.create_table(
        "extraction_prompt_activation",
        sa.Column("activation_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version_hash", sa.Text(), nullable=False),
        sa.Column("previous_version_hash", sa.Text(), nullable=True),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("is_rollback", sa.Boolean(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("activation_id", name="pk_extraction_prompt_activation"),
        # Activating a hash nobody saved would leave the task with no resolvable
        # prompt at its next call, and the failure would surface far from the
        # mistake. The FK is the guard, not a decoration.
        sa.ForeignKeyConstraint(
            ("name", "version_hash"),
            ("extraction_prompt_version.name", "extraction_prompt_version.version_hash"),
            name="fk_extraction_prompt_activation_version",
        ),
        sa.UniqueConstraint("name", "sequence", name="uq_extraction_prompt_activation_sequence"),
        sa.CheckConstraint("name <> ''", name="name_not_empty"),
        sa.CheckConstraint("version_hash <> ''", name="version_hash_not_empty"),
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        sa.CheckConstraint("sequence >= 1", name="sequence_positive"),
        # A prompt's first activation has no predecessor and every later one
        # does. A NULL predecessor on a later activation would silently break
        # the chain a reader walks to reconstruct what was in force when.
        sa.CheckConstraint(
            "(sequence = 1) = (previous_version_hash IS NULL)",
            name="first_activation_no_predecessor",
        ),
    )
    op.create_table(
        "extraction_golden_score",
        sa.Column("score_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version_hash", sa.Text(), nullable=False),
        sa.Column("golden_set_id", sa.Text(), nullable=False),
        sa.Column("agreement", sa.Numeric(6, 5), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("scored_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("score_id", name="pk_extraction_golden_score"),
        sa.ForeignKeyConstraint(
            ("name", "version_hash"),
            ("extraction_prompt_version.name", "extraction_prompt_version.version_hash"),
            name="fk_extraction_golden_score_version",
        ),
        sa.CheckConstraint("name <> ''", name="name_not_empty"),
        sa.CheckConstraint("version_hash <> ''", name="version_hash_not_empty"),
        sa.CheckConstraint("golden_set_id <> ''", name="golden_set_id_not_empty"),
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        # Agreement is a FRACTION, never a percentage (§8). The bound is the
        # unit check: a caller passing 85 fails here rather than storing a
        # number that reads as a triumph.
        sa.CheckConstraint("agreement >= 0 AND agreement <= 1", name="agreement_is_a_fraction"),
        sa.CheckConstraint("document_count >= 1", name="document_count_positive"),
    )
    op.create_index(
        "ix_extraction_golden_score_version",
        "extraction_golden_score",
        ["name", "version_hash"],
    )
    op.create_table(
        "extraction_result",
        sa.Column("result_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("prompt_version_hash", sa.Text(), nullable=False),
        sa.Column("payload_digest", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=False),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "validation_errors",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Numeric(12, 3), nullable=True),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column(
            "extracted_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("result_id", name="pk_extraction_result"),
        # No FK to extraction_prompt_version on purpose: a prompt is addressable
        # whether or not anyone chose to save it to the library, and refusing to
        # record an extraction because its prompt was unsaved would discard the
        # observation in order to protect a join.
        sa.CheckConstraint("task <> ''", name="task_not_empty"),
        sa.CheckConstraint("document_id <> ''", name="document_id_not_empty"),
        sa.CheckConstraint("prompt_version_hash <> ''", name="prompt_version_hash_not_empty"),
        sa.CheckConstraint("payload_digest <> ''", name="payload_digest_not_empty"),
        sa.CheckConstraint("model <> ''", name="model_not_empty"),
        sa.CheckConstraint("chunk_count >= 1", name="chunk_count_positive"),
        sa.CheckConstraint(
            "chunk_index >= 0 AND chunk_index < chunk_count", name="chunk_index_in_range"
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_counted"
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_counted"
        ),
        sa.CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_non_negative"),
        # A response either validated or it did not. Two independently nullable
        # columns would permit "an output with errors beside it" and "neither",
        # both of which are records nobody could act on.
        sa.CheckConstraint(
            "(output IS NOT NULL) <> (jsonb_array_length(validation_errors) > 0)",
            name="output_xor_validation_errors",
        ),
    )
    op.create_index(
        "ix_extraction_result_document",
        "extraction_result",
        ["task", "document_id"],
    )
    op.create_index(
        "ix_extraction_result_prompt_version",
        "extraction_result",
        ["prompt_version_hash"],
    )
    op.execute(
        """
        CREATE FUNCTION extraction_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'extraction table % is append-only: % rejected; a prompt version, an '
                'activation, a golden-set score and an extraction result are all '
                'observations, and an observation that can be edited afterwards is not '
                'evidence (DIRECTIVE 5-P7 and 6.5, P7.3/P7.10)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_extraction_prompt_version_append_only "
        "BEFORE UPDATE OR DELETE ON extraction_prompt_version "
        "FOR EACH ROW EXECUTE FUNCTION extraction_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_extraction_prompt_activation_append_only "
        "BEFORE UPDATE OR DELETE ON extraction_prompt_activation "
        "FOR EACH ROW EXECUTE FUNCTION extraction_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_extraction_golden_score_append_only "
        "BEFORE UPDATE OR DELETE ON extraction_golden_score "
        "FOR EACH ROW EXECUTE FUNCTION extraction_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_extraction_result_append_only "
        "BEFORE UPDATE OR DELETE ON extraction_result "
        "FOR EACH ROW EXECUTE FUNCTION extraction_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the append-only triggers, their function, and all four tables."""
    for table in reversed(_APPEND_ONLY_TABLES):
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.execute("DROP FUNCTION extraction_append_only_guard()")
    op.drop_index("ix_extraction_result_prompt_version", table_name="extraction_result")
    op.drop_index("ix_extraction_result_document", table_name="extraction_result")
    op.drop_table("extraction_result")
    op.drop_index("ix_extraction_golden_score_version", table_name="extraction_golden_score")
    op.drop_table("extraction_golden_score")
    op.drop_table("extraction_prompt_activation")
    op.drop_table("extraction_prompt_version")
