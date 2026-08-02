"""Alerts, their delivery and acknowledgement, and the monitoring halt history (P12.1/P12.4).

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-02

Four tables, none bitemporal, all **append-only** by the same
``BEFORE UPDATE OR DELETE`` row-trigger shape revisions
0003/0004/0007/0009/0010/0011/0013/0014 use:

- ``monitoring_alert`` — one row per alert *condition*, identified by a
  content-derived ``dedup_key``;
- ``monitoring_alert_delivery`` — one row per attempt to hand an alert to a
  channel, failures included;
- ``monitoring_alert_acknowledgement`` — one row per alert, naming the person who
  signed for it;
- ``monitoring_halt_event`` — the halt/resume history, from which the current
  halt state is derived.

Four enforcement points the application cannot waive
----------------------------------------------------

**1. One alert per condition: ``UNIQUE (dedup_key)``.** The key is a SHA-256 over
the rule and the condition (``backend.monitoring.alerts.alert_dedup_key``).
Checking for an existing alert in Python would be a read followed by a write with
a window between them, and two monitoring workers racing through that window both
find the key free and both page the operator for one condition. The constraint
has no window. A CHECK additionally requires the key to *look* like a digest,
because a key that is not derived from content is a counter wearing a digest's
clothes — and a counter has to be remembered across the restart it exists for.

**2. One acknowledgement per alert: ``UNIQUE (alert_id)``.** An acknowledgement
records *who took responsibility*. A second one is refused rather than
overwriting the first, because a silently replaced name still looks like a
complete record.

**3. One resume per halt: ``UNIQUE (resolves_halt_event_id)``.** The current halt
state is *derived* — the earliest ``halt`` row no ``resume`` row points at — so
two resumes of one halt would make the derived state ambiguous. NULLs are
distinct in Postgres, so this constrains resumes only and leaves halt rows
completely unconstrained. That asymmetry is deliberate and matches 0016's
reasoning about ``execution_halt``: a constraint that can reject a *halt* row is
a constraint that can stop the halt from being recorded, and a redundant halt row
costs nothing while a refused one costs everything.

**4. The two halt-row shapes, in SQL.** ``shape_matches_kind`` binds a halt to
"has a cause, resolves nothing" and a resume to "resolves exactly one halt,
carries no cause". A writer that skips ``backend.monitoring.history`` entirely is
still bound.

No state columns anywhere
-------------------------

``monitoring_alert`` has no ``delivered`` and no ``acknowledged``;
``monitoring_halt_event`` has no ``is_active``. All three are derived from the
rows. An append-only table cannot update a denormalized flag, and a flag that
drifts from the history explaining it is the worst possible field here — it is
the one an operator would trust most. Same reasoning as ``execution_order``'s
absent state column (D-033).

Not bitemporal
--------------

Same reasoning as revisions 0005, 0007, 0009, 0010, 0011, 0013 and 0014. The
bitemporal columns describe when a fact was true in the world and when it became
knowable to the *market* (D-011). An alert and a halt are things **we** observed
about our own system; a ``knowledge_time`` invented for them would be a
fabricated value in the one column whose meaning is that it is not fabricated
(I3). So these tables are absent from the bitemporal registry, unscoped by the
Core-level read guard, and read without an as-of.

TRUNCATE stays unblocked on all four, matching every earlier revision: it is the
sanctioned admin and test reset path and never masquerades as an edit.

Downgrade drops the triggers, their function, the indices and the tables in
dependency order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = (
    "monitoring_alert",
    "monitoring_alert_delivery",
    "monitoring_alert_acknowledgement",
    "monitoring_halt_event",
)
"""Tables that get the append-only trigger, in creation order.

Named here *and* spelled out in full ``CREATE TRIGGER`` statements below rather
than generated from this tuple. The duplication is deliberate: "which tables in
this schema are append-only" is a question a human answers by grepping for
``BEFORE UPDATE OR DELETE ON``, and a loop that assembles the statement makes
that grep return nothing. Revisions 0003, 0004, 0007, 0009, 0010, 0011, 0013 and
0014 write theirs out for the same reason.
"""

_HALT_CAUSES_SQL = (
    "'below_expected_band', 'above_expected_band', 'comparison_unavailable', "
    "'stale_live_data', 'cost_basis', 'internal_error'"
)
"""The members of ``backend.monitoring.expectation.HaltCause``, as SQL literals.

Enumerated rather than left free text because each value names a *different
operator response*: "the comparison could not be made" and "live performance was
below the band" are not the same incident, and a halt history that cannot be
filtered by cause is a list of outages nobody can reason about. A test asserts
this list against the Python enum so the two cannot drift.
"""


def upgrade() -> None:
    """Create the four monitoring tables, their constraints, indices and triggers."""
    op.create_table(
        "monitoring_alert",
        sa.Column("alert_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("dedup_key", sa.Text(), nullable=False),
        sa.Column("rule_id", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("raised_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("git_commit", sa.Text(), nullable=False),
        sa.Column("git_dirty", sa.Boolean(), nullable=False),
        sa.Column("data_version", sa.Text(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("alert_id", name="pk_monitoring_alert"),
        # The mechanism, not an optimisation: one condition, one alert, one
        # acknowledgement, whatever else is running.
        sa.UniqueConstraint("dedup_key", name="uq_monitoring_alert_dedup_key"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005, 0007, 0009-0014).
        sa.CheckConstraint("severity IN ('info', 'warning', 'critical')", name="severity_known"),
        sa.CheckConstraint("dedup_key ~ '^[0-9a-f]{64}$'", name="dedup_key_is_digest"),
        sa.CheckConstraint("rule_id <> ''", name="rule_id_not_empty"),
        sa.CheckConstraint("subject <> ''", name="subject_not_empty"),
        sa.CheckConstraint("detail <> ''", name="detail_not_empty"),
        sa.CheckConstraint(
            "git_commit ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'", name="git_commit_is_sha"
        ),
        sa.CheckConstraint("data_version <> ''", name="data_version_not_empty"),
        sa.CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        sa.CheckConstraint("seed >= 0", name="seed_non_negative"),
    )
    op.create_table(
        "monitoring_alert_delivery",
        sa.Column("delivery_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("alert_id", sa.BigInteger(), nullable=False),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("attempted_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("delivery_id", name="pk_monitoring_alert_delivery"),
        sa.ForeignKeyConstraint(
            ["alert_id"],
            ["monitoring_alert.alert_id"],
            name="fk_monitoring_alert_delivery_alert_id_monitoring_alert",
            ondelete="RESTRICT",
        ),
        # Monotonic per channel, and a retry cannot overwrite the record of the
        # attempt it is retrying.
        sa.UniqueConstraint(
            "alert_id", "channel_id", "attempt", name="uq_monitoring_alert_delivery_attempt"
        ),
        sa.CheckConstraint("outcome IN ('delivered', 'failed')", name="outcome_known"),
        sa.CheckConstraint("attempt >= 1", name="attempt_positive"),
        sa.CheckConstraint("channel_id <> ''", name="channel_id_not_empty"),
        # A failure that does not say why cannot be acted on: the response to
        # "SMTP refused the recipient" is not the response to "credential expired".
        sa.CheckConstraint("detail <> ''", name="detail_not_empty"),
    )
    op.create_table(
        "monitoring_alert_acknowledgement",
        sa.Column("acknowledgement_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("alert_id", sa.BigInteger(), nullable=False),
        sa.Column("acknowledged_by", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("acknowledged_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("acknowledgement_id", name="pk_monitoring_alert_acknowledgement"),
        sa.ForeignKeyConstraint(
            ["alert_id"],
            ["monitoring_alert.alert_id"],
            name="fk_monitoring_alert_acknowledgement_alert_id_monitoring_alert",
            ondelete="RESTRICT",
        ),
        # One signature per alert, never replaced.
        sa.UniqueConstraint("alert_id", name="uq_monitoring_alert_acknowledgement_alert"),
        sa.CheckConstraint("acknowledged_by <> ''", name="acknowledged_by_not_empty"),
        # An acknowledgement with no note is a click; the point of the record is
        # that somebody looked and concluded something.
        sa.CheckConstraint("note <> ''", name="note_not_empty"),
    )
    op.create_table(
        "monitoring_halt_event",
        sa.Column("halt_event_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("cause", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("resolves_halt_event_id", sa.BigInteger(), nullable=True),
        sa.Column("alert_dedup_key", sa.Text(), nullable=True),
        sa.Column("decision", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("git_commit", sa.Text(), nullable=False),
        sa.Column("git_dirty", sa.Boolean(), nullable=False),
        sa.Column("data_version", sa.Text(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("halt_event_id", name="pk_monitoring_halt_event"),
        sa.ForeignKeyConstraint(
            ["resolves_halt_event_id"],
            ["monitoring_halt_event.halt_event_id"],
            name="fk_monitoring_halt_event_resolves_halt_event_id_monitoring_halt_event",
            ondelete="RESTRICT",
        ),
        # A halt is resumed at most once. NULLs are distinct in Postgres, so halt
        # rows — which all carry NULL here — stay completely unconstrained: a
        # constraint able to refuse a halt row could stop a halt being recorded.
        sa.UniqueConstraint("resolves_halt_event_id", name="uq_monitoring_halt_event_resolves"),
        sa.CheckConstraint("kind IN ('halt', 'resume')", name="kind_known"),
        sa.CheckConstraint("detail <> ''", name="detail_not_empty"),
        # 'who halted trading' has an answer even when the answer is a scheduled
        # job ('auto:monitoring'); NULL there would read as 'nobody knows'.
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        sa.CheckConstraint(
            "(kind = 'halt' AND cause IS NOT NULL AND resolves_halt_event_id IS NULL) "
            "OR (kind = 'resume' AND cause IS NULL AND resolves_halt_event_id IS NOT NULL)",
            name="shape_matches_kind",
        ),
        sa.CheckConstraint(
            f"cause IS NULL OR cause IN ({_HALT_CAUSES_SQL})",
            name="cause_known",
        ),
        sa.CheckConstraint(
            "git_commit ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'", name="git_commit_is_sha"
        ),
        sa.CheckConstraint("data_version <> ''", name="data_version_not_empty"),
        sa.CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        sa.CheckConstraint("seed >= 0", name="seed_non_negative"),
    )

    # The read paths, and the only ones. "Which alerts has nobody signed for" is
    # on the operator's screen and runs on every dashboard poll; the delivery and
    # acknowledgement lookups run once per alert per dispatch.
    op.create_index(
        "ix_monitoring_alert_severity_raised_at",
        "monitoring_alert",
        ["severity", sa.text("raised_at DESC")],
    )
    op.create_index(
        "ix_monitoring_alert_delivery_alert",
        "monitoring_alert_delivery",
        ["alert_id", "channel_id"],
    )
    # Partial and deliberate: "which halts are still open" is the question the
    # kill-switch seam asks on every cycle, and a resume row is never its answer.
    op.create_index(
        "ix_monitoring_halt_event_open",
        "monitoring_halt_event",
        ["halt_event_id"],
        postgresql_where=sa.text("kind = 'halt'"),
    )
    op.create_index(
        "ix_monitoring_halt_event_resolves",
        "monitoring_halt_event",
        ["resolves_halt_event_id"],
        postgresql_where=sa.text("resolves_halt_event_id IS NOT NULL"),
    )

    op.execute(
        """
        CREATE FUNCTION monitoring_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'monitoring table % is append-only (P12.4): % rejected. An alert records a '
                'condition that was observed, an acknowledgement records that a person took '
                'responsibility for it, and a halt records that trading stopped — none of '
                'them stops being true later. A correction is a new row, and erasing one '
                'would destroy the incident trail a halt has to be reviewed through '
                '(DIRECTIVE 6.10, 6.11, I2)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_monitoring_alert_append_only "
        "BEFORE UPDATE OR DELETE ON monitoring_alert "
        "FOR EACH ROW EXECUTE FUNCTION monitoring_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_monitoring_alert_delivery_append_only "
        "BEFORE UPDATE OR DELETE ON monitoring_alert_delivery "
        "FOR EACH ROW EXECUTE FUNCTION monitoring_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_monitoring_alert_acknowledgement_append_only "
        "BEFORE UPDATE OR DELETE ON monitoring_alert_acknowledgement "
        "FOR EACH ROW EXECUTE FUNCTION monitoring_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_monitoring_halt_event_append_only "
        "BEFORE UPDATE OR DELETE ON monitoring_halt_event "
        "FOR EACH ROW EXECUTE FUNCTION monitoring_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the triggers, their function, the indices and the four tables."""
    for table in reversed(_APPEND_ONLY_TABLES):
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.execute("DROP FUNCTION monitoring_append_only_guard()")
    op.drop_index("ix_monitoring_halt_event_resolves", table_name="monitoring_halt_event")
    op.drop_index("ix_monitoring_halt_event_open", table_name="monitoring_halt_event")
    op.drop_index("ix_monitoring_alert_delivery_alert", table_name="monitoring_alert_delivery")
    op.drop_index("ix_monitoring_alert_severity_raised_at", table_name="monitoring_alert")
    op.drop_table("monitoring_halt_event")
    op.drop_table("monitoring_alert_acknowledgement")
    op.drop_table("monitoring_alert_delivery")
    op.drop_table("monitoring_alert")
