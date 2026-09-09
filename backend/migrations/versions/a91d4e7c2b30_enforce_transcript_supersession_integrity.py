"""enforce transcript supersession integrity

Revision ID: a91d4e7c2b30
Revises: f3b9c7d1a620
Create Date: 2026-07-24 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "a91d4e7c2b30"
down_revision: Union[str, None] = "f3b9c7d1a620"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    # Clear any invalid relationship written before the invariant existed.
    # Anonymous history may safely become attributed, but attributed history
    # cannot move to another operator or back to anonymous.
    op.execute(
        """
        UPDATE transcripts AS child
        SET supersedes_transcript_id = NULL
        FROM transcripts AS parent
        WHERE child.supersedes_transcript_id = parent.id
          AND (
            child.call_id <> parent.call_id
            OR child.recording_id <> parent.recording_id
            OR (
              parent.operator_id IS NOT NULL
              AND parent.operator_id IS DISTINCT FROM child.operator_id
            )
          )
        """
    )
    # A corrupt cycle has no trustworthy root. Break every edge whose lineage
    # reaches that cycle; the transcript rows and their content remain intact.
    op.execute(
        """
        WITH RECURSIVE lineage(origin_id, current_id, next_id, path, has_cycle) AS (
          SELECT
            transcript.id,
            transcript.id,
            transcript.supersedes_transcript_id,
            ARRAY[transcript.id],
            FALSE
          FROM transcripts AS transcript
          WHERE transcript.supersedes_transcript_id IS NOT NULL

          UNION ALL

          SELECT
            lineage.origin_id,
            parent.id,
            parent.supersedes_transcript_id,
            lineage.path || parent.id,
            parent.id = ANY(lineage.path)
          FROM lineage
          JOIN transcripts AS parent ON parent.id = lineage.next_id
          WHERE lineage.next_id IS NOT NULL
            AND NOT lineage.has_cycle
        ),
        unsafe_origins AS (
          SELECT DISTINCT origin_id
          FROM lineage
          WHERE has_cycle
        )
        UPDATE transcripts
        SET supersedes_transcript_id = NULL
        WHERE id IN (SELECT origin_id FROM unsafe_origins)
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_transcript_supersession_integrity()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
          has_cycle boolean;
          has_invalid_target boolean;
        BEGIN
          -- Serialize every lineage mutation for a recording. Without this
          -- shared lock, concurrent A->B and B->A updates could each validate
          -- against the other's old state and commit a cycle.
          IF TG_OP = 'UPDATE' THEN
            PERFORM recording.id
            FROM recordings AS recording
            WHERE recording.id IN (OLD.recording_id, NEW.recording_id)
            ORDER BY recording.id
            FOR UPDATE;
          ELSE
            PERFORM recording.id
            FROM recordings AS recording
            WHERE recording.id = NEW.recording_id
            FOR UPDATE;
          END IF;

          -- A transcript's call/recording identity is immutable. Anonymous
          -- history may acquire an operator through PBX/manual attribution,
          -- but an already-attributed row must retain that operator so a leaf
          -- cannot be retargeted without creating an explicit replacement.
          IF TG_OP = 'UPDATE' AND (
            OLD.call_id IS DISTINCT FROM NEW.call_id
            OR OLD.recording_id IS DISTINCT FROM NEW.recording_id
            OR (
              OLD.operator_id IS NOT NULL
              AND OLD.operator_id IS DISTINCT FROM NEW.operator_id
            )
          ) THEN
            RAISE EXCEPTION
              'transcript update crosses an immutable logical target boundary'
              USING ERRCODE = '23514';
          END IF;

          IF NEW.supersedes_transcript_id IS NOT NULL THEN
            WITH RECURSIVE lineage(
              id,
              call_id,
              recording_id,
              operator_id,
              supersedes_transcript_id,
              path,
              cycle
            ) AS (
              SELECT
                transcript.id,
                transcript.call_id,
                transcript.recording_id,
                transcript.operator_id,
                transcript.supersedes_transcript_id,
                ARRAY[transcript.id],
                transcript.id = NEW.id
              FROM transcripts AS transcript
              WHERE transcript.id = NEW.supersedes_transcript_id

              UNION ALL

              SELECT
                parent.id,
                parent.call_id,
                parent.recording_id,
                parent.operator_id,
                parent.supersedes_transcript_id,
                lineage.path || parent.id,
                parent.id = NEW.id OR parent.id = ANY(lineage.path)
              FROM lineage
              JOIN transcripts AS parent
                ON parent.id = lineage.supersedes_transcript_id
              WHERE lineage.supersedes_transcript_id IS NOT NULL
                AND NOT lineage.cycle
            )
            SELECT
              COALESCE(bool_or(id = NEW.id OR cycle), FALSE),
              COALESCE(
                bool_or(
                  call_id <> NEW.call_id
                  OR recording_id <> NEW.recording_id
                  OR (
                    operator_id IS NOT NULL
                    AND operator_id IS DISTINCT FROM NEW.operator_id
                  )
                ),
                FALSE
              )
            INTO has_cycle, has_invalid_target
            FROM lineage;

            IF has_cycle THEN
              RAISE EXCEPTION
                'transcript supersession would create a cycle'
                USING ERRCODE = '23514';
            END IF;
            IF has_invalid_target THEN
              RAISE EXCEPTION
                'transcript supersession crosses a logical target boundary'
                USING ERRCODE = '23514';
            END IF;
          END IF;

          IF EXISTS (
            SELECT 1
            FROM transcripts AS child
            WHERE child.supersedes_transcript_id = NEW.id
              AND (
                child.call_id <> NEW.call_id
                OR child.recording_id <> NEW.recording_id
                OR (
                  NEW.operator_id IS NOT NULL
                  AND NEW.operator_id IS DISTINCT FROM child.operator_id
                )
              )
          ) THEN
            RAISE EXCEPTION
              'transcript update would invalidate a superseding child'
              USING ERRCODE = '23514';
          END IF;

          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_transcripts_supersession_integrity
        BEFORE INSERT OR UPDATE OF
          supersedes_transcript_id,
          call_id,
          recording_id,
          operator_id
        ON transcripts
        FOR EACH ROW
        EXECUTE FUNCTION enforce_transcript_supersession_integrity()
        """
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TRIGGER IF EXISTS trg_transcripts_supersession_integrity ON transcripts")
    op.execute("DROP FUNCTION IF EXISTS enforce_transcript_supersession_integrity()")
