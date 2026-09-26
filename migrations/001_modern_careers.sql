-- P03-A / migration 001: modern identity storage only; no lifecycle writer.
-- Explicit operator invocation only, e.g. psql -X -v ON_ERROR_STOP=1 -f this_file.
-- Never run from startup, auth, Cloud, health or capabilities.
-- Dry-run: execute the statements in a test transaction and ROLLBACK.
-- Backend rollback preserves this table; no destructive down migration.
BEGIN;
SELECT pg_advisory_xact_lock(503, 1);

CREATE TABLE IF NOT EXISTS public.modern_careers (
    user_id TEXT NOT NULL,
    profile_uuid UUID NOT NULL,
    career_id UUID NOT NULL,
    generation BIGINT NOT NULL DEFAULT 1,
    state TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ NULL,
    CONSTRAINT modern_careers_pk PRIMARY KEY (career_id),
    CONSTRAINT modern_careers_user_ck CHECK (user_id <> '' AND user_id = btrim(user_id)),
    CONSTRAINT modern_careers_generation_ck CHECK (generation >= 1),
    CONSTRAINT modern_careers_state_ck CHECK (state IN ('ACTIVE', 'DELETED')),
    CONSTRAINT modern_careers_deleted_ck CHECK (
        (state = 'ACTIVE' AND deleted_at IS NULL)
        OR (state = 'DELETED' AND deleted_at IS NOT NULL)
    )
);

-- Owner/profile listing and owner-filtered lookup; PK handles global identity.
CREATE INDEX IF NOT EXISTS modern_careers_owner_idx
    ON public.modern_careers (user_id, profile_uuid);

-- Fail closed on incompatible pre-existing objects; replay is not a repair.
-- PostgreSQL 18 catalog definitions are checked without reading legacy tables.
DO $migration$
DECLARE
    actual JSONB;
BEGIN
    SELECT jsonb_agg(jsonb_build_array(a.attname,
               format_type(a.atttypid, a.atttypmod), a.attnotnull,
               pg_get_expr(d.adbin, d.adrelid)) ORDER BY a.attnum)
      INTO actual
      FROM pg_attribute a
      LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
     WHERE a.attrelid = 'public.modern_careers'::regclass
       AND a.attnum > 0 AND NOT a.attisdropped;
    IF actual IS DISTINCT FROM '[
        ["user_id", "text", true, null],
        ["profile_uuid", "uuid", true, null],
        ["career_id", "uuid", true, null],
        ["generation", "bigint", true, "1"],
        ["state", "text", true, "''ACTIVE''::text"],
        ["created_at", "timestamp with time zone", true, "now()"],
        ["updated_at", "timestamp with time zone", true, "now()"],
        ["deleted_at", "timestamp with time zone", false, null]
    ]'::jsonb THEN
        RAISE EXCEPTION 'P03_001_SCHEMA_MISMATCH: columns/defaults';
    END IF;

    SELECT jsonb_object_agg(conname, pg_get_constraintdef(oid))
      INTO actual FROM pg_constraint
     WHERE conrelid = 'public.modern_careers'::regclass
       AND convalidated AND contype <> 'n'; -- NOT NULL checked via attnotnull above
    IF actual IS DISTINCT FROM jsonb_build_object(
        'modern_careers_pk', 'PRIMARY KEY (career_id)',
        'modern_careers_user_ck', 'CHECK (((user_id <> ''''::text) AND (user_id = btrim(user_id))))',
        'modern_careers_generation_ck', 'CHECK ((generation >= 1))',
        'modern_careers_state_ck', 'CHECK ((state = ANY (ARRAY[''ACTIVE''::text, ''DELETED''::text])))',
        'modern_careers_deleted_ck', 'CHECK ((((state = ''ACTIVE''::text) AND (deleted_at IS NULL)) OR ((state = ''DELETED''::text) AND (deleted_at IS NOT NULL))))'
    ) OR EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'public.modern_careers'::regclass AND NOT convalidated
    ) THEN
        RAISE EXCEPTION 'P03_001_SCHEMA_MISMATCH: constraints';
    END IF;

    SELECT jsonb_object_agg(indexname, indexdef) INTO actual
      FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'modern_careers';
    IF actual IS DISTINCT FROM jsonb_build_object(
        'modern_careers_pk', 'CREATE UNIQUE INDEX modern_careers_pk ON public.modern_careers USING btree (career_id)',
        'modern_careers_owner_idx', 'CREATE INDEX modern_careers_owner_idx ON public.modern_careers USING btree (user_id, profile_uuid)'
    ) THEN
        RAISE EXCEPTION 'P03_001_SCHEMA_MISMATCH: indexes';
    END IF;
END
$migration$;
COMMIT;
