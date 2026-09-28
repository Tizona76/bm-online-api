-- Explicit migration only; never execute at startup. PostgreSQL 15+ (target: 18).
-- Requires migration 004. Preserves every row, including at most one NULL slot.
-- Pause Cloud traffic while applying 005 and deploying the matching writer.
-- The old ON CONFLICT(user_id, profile_uuid) writer is incompatible afterwards.
-- Do not roll back to profile-only uniqueness once multiple careers exist.
BEGIN;
LOCK TABLE public.cloud_saves_v2 IN ACCESS EXCLUSIVE MODE;
DO $migration$
DECLARE
    slot_oid oid := 'public.cloud_saves_v2'::regclass;
    column_count integer;
    constraint_def text;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE oid = slot_oid AND relkind = 'r')
       OR EXISTS (
        SELECT 1 FROM (VALUES
            ('save_id', 'text', true), ('user_id', 'text', true),
            ('profile_uuid', 'text', true), ('rev', 'integer', true),
            ('checksum', 'text', false), ('blob_json', 'jsonb', true),
            ('blob_size', 'integer', true), ('updated_at', 'timestamp with time zone', true)
        ) AS expected(name, type_name, required)
        LEFT JOIN pg_attribute a ON a.attrelid = slot_oid AND a.attname = expected.name
                                AND a.attnum > 0 AND NOT a.attisdropped
        WHERE a.attname IS NULL OR format_type(a.atttypid, a.atttypmod) <> expected.type_name
           OR a.attnotnull <> expected.required OR a.attgenerated <> '' OR a.attidentity <> ''
    ) THEN
        RAISE EXCEPTION 'cloud_saves_v2: incompatible base schema';
    END IF;
    SELECT count(*) INTO column_count FROM pg_attribute
    WHERE attrelid = slot_oid AND attname = 'career_id' AND attnum > 0 AND NOT attisdropped;
    IF column_count = 0 THEN
        RAISE EXCEPTION 'cloud_saves_v2: apply migration 004 first';
    ELSE
        IF NOT EXISTS (
            SELECT 1 FROM pg_attribute a
            WHERE a.attrelid = slot_oid AND a.attname = 'career_id'
              AND a.atttypid = 'text'::regtype AND a.atttypmod = -1
              AND NOT a.attnotnull AND NOT a.atthasdef
              AND a.attgenerated = '' AND a.attidentity = ''
              AND a.attcollation = (SELECT typcollation FROM pg_type WHERE oid = 'text'::regtype)
        ) THEN
            RAISE EXCEPTION 'cloud_saves_v2: incompatible career_id column';
        END IF;
    END IF;
    SELECT pg_get_constraintdef(c.oid) INTO constraint_def FROM pg_constraint c
    WHERE c.conrelid = slot_oid AND c.conname = 'cloud_saves_v2_career_id_check'
      AND c.contype = 'c' AND c.convalidated AND NOT c.connoinherit;
    IF constraint_def IS DISTINCT FROM
        'CHECK (((career_id IS NULL) OR (((length(career_id) >= 1) AND (length(career_id) <= 128)) AND (career_id = btrim(career_id)))))'
    THEN
        RAISE EXCEPTION 'cloud_saves_v2: incompatible career_id constraint: %', constraint_def;
    END IF;
    SELECT pg_get_constraintdef(c.oid) INTO constraint_def FROM pg_constraint c
    WHERE c.conrelid = slot_oid AND c.conname = 'cloud_saves_v2_uq'
      AND c.contype = 'u' AND NOT c.condeferrable AND c.convalidated;
    IF constraint_def = 'UNIQUE (user_id, profile_uuid)' THEN
        ALTER TABLE public.cloud_saves_v2 DROP CONSTRAINT cloud_saves_v2_uq;
        ALTER TABLE public.cloud_saves_v2 ADD CONSTRAINT cloud_saves_v2_uq
            UNIQUE NULLS NOT DISTINCT (user_id, profile_uuid, career_id);
    ELSIF constraint_def IS DISTINCT FROM
        'UNIQUE NULLS NOT DISTINCT (user_id, profile_uuid, career_id)' THEN
        RAISE EXCEPTION 'cloud_saves_v2: incompatible slot uniqueness: %', constraint_def;
    END IF;
    -- A remaining unique profile-only index would still prevent a second career.
    IF EXISTS (
        SELECT 1 FROM pg_index i WHERE i.indrelid = slot_oid AND i.indisunique
          AND i.indnkeyatts = 2 AND i.indexprs IS NULL
          AND i.indkey[0] IN (
            SELECT attnum FROM pg_attribute WHERE attrelid = slot_oid AND attname IN ('user_id', 'profile_uuid'))
          AND i.indkey[1] IN (
            SELECT attnum FROM pg_attribute WHERE attrelid = slot_oid AND attname IN ('user_id', 'profile_uuid'))
    ) THEN
        RAISE EXCEPTION 'cloud_saves_v2: redundant profile-only unique index';
    END IF;
END
$migration$;
COMMIT;
