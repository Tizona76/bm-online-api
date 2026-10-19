-- Explicit migration only; never run at backend startup.
-- Ordered prerequisites: users -> unchanged A3 -> this A5 script -> A4 script.
-- A4 installs and validates steam_wallet_a4_receipt_fk. Keep routes disabled
-- until BOTH scripts succeed. This script alone is not a complete Steam schema.
-- Reexecution preserves rows; existing incompatible schemas require review.
-- Run as the migration owner. No privileges are granted to PUBLIC or new roles.
BEGIN;
CREATE TABLE IF NOT EXISTS steam_wallet_checkpoints (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    profile_uuid TEXT NOT NULL,
    career_id TEXT NOT NULL,
    revision BIGINT NOT NULL CHECK (revision BETWEEN 1 AND 9007199254740991),
    -- Canonical JSON stored as text so hashes describe exact server serialization.
    snapshot TEXT NOT NULL CHECK (jsonb_typeof(snapshot::jsonb) = 'object'),
    checkpoint_hash TEXT NOT NULL CHECK (checkpoint_hash ~ '^[0-9a-f]{64}$'),
    last_key TEXT,
    last_request_hash TEXT,
    PRIMARY KEY (user_id, profile_uuid, career_id),
    FOREIGN KEY (user_id) REFERENCES steam_identity_links(user_id),
    CHECK ((last_key IS NULL) = (last_request_hash IS NULL))
);
CREATE TABLE IF NOT EXISTS steam_wallet_deliveries (
    operation_id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL UNIQUE, -- A4 adds steam_wallet_a4_receipt_fk after its receipt table exists.
    user_id TEXT NOT NULL,
    profile_uuid TEXT NOT NULL,
    career_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL CHECK (request_hash ~ '^[0-9a-f]{64}$'),
    receipt_hash TEXT NOT NULL CHECK (receipt_hash ~ '^[0-9a-f]{64}$'),
    expected_revision BIGINT NOT NULL CHECK (expected_revision > 0),
    parent_hash TEXT NOT NULL CHECK (parent_hash ~ '^[0-9a-f]{64}$'),
    candidate TEXT NOT NULL CHECK (jsonb_typeof(candidate::jsonb) = 'object'),
    candidate_hash TEXT NOT NULL CHECK (candidate_hash ~ '^[0-9a-f]{64}$'),
    state TEXT NOT NULL CHECK (state IN ('prepared', 'acknowledged')),
    UNIQUE (user_id, idempotency_key),
    UNIQUE (operation_id, candidate_hash),
    FOREIGN KEY (user_id, profile_uuid, career_id)
        REFERENCES steam_wallet_checkpoints(user_id, profile_uuid, career_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS steam_wallet_one_prepared_career
    ON steam_wallet_deliveries(user_id, profile_uuid, career_id)
    WHERE state = 'prepared';
CREATE TABLE IF NOT EXISTS steam_wallet_acks (
    operation_id TEXT PRIMARY KEY,
    revision BIGINT NOT NULL CHECK (revision > 0),
    candidate_hash TEXT NOT NULL,
    FOREIGN KEY (operation_id, candidate_hash)
        REFERENCES steam_wallet_deliveries(operation_id, candidate_hash)
);
CREATE OR REPLACE FUNCTION steam_wallet_delivery_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW) - 'state') IS DISTINCT FROM (to_jsonb(OLD) - 'state')
       OR (OLD.state = 'acknowledged' AND NEW.state <> OLD.state) THEN
        RAISE EXCEPTION 'IMMUTABLE_STEAM_DELIVERY' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER steam_wallet_delivery_immutable
    BEFORE UPDATE ON steam_wallet_deliveries
    FOR EACH ROW EXECUTE FUNCTION steam_wallet_delivery_immutable();
-- A first A5 installation precedes A4; a replay after A4 must not hide a
-- missing/unvalidated cross-registry reference. A4 validates the exact columns.
DO $$ BEGIN
    IF to_regclass('steam_receipts') IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conrelid='steam_wallet_deliveries'::regclass
          AND conname='steam_wallet_a4_receipt_fk' AND contype='f'
          AND confrelid=to_regclass('steam_receipts') AND convalidated
    ) THEN
        RAISE EXCEPTION 'A5_MIGRATION_BLOCKED_PENDING_A4_RECEIPT_FOREIGN_KEY';
    END IF;
END $$;
COMMIT;
-- Rollback policy: disable STEAM_WALLET_ENABLED, preserve all rows/receipts.
-- No DROP, truncation, cascade or automatic deletion is part of this migration.
