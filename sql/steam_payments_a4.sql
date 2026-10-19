-- Explicit migration only; no startup DDL, no automatic production execution.
-- Order: users -> unchanged A3 -> steam_wallet_a5.sql -> this A4 script.
-- A5 tables must already exist. This script completes the A5 receipt foreign key.
-- Reexecution preserves orders, receipts and reconciliation history.
-- Rollback: disable feature flags and preserve all rows; never DROP receipts.
-- Owner executes DDL; provision the existing backend role separately, without
-- broad PUBLIC grants. Verify the actual Render role/schema before deployment.
BEGIN;
-- BEGIN TABLE DDL (tests use only this section in a private synthetic schema)
CREATE TABLE IF NOT EXISTS steam_orders (
    order_id BIGINT PRIMARY KEY CHECK(order_id>0),
    user_id TEXT NOT NULL REFERENCES steam_identity_links(user_id),
    profile_uuid TEXT NOT NULL,
    career_id TEXT NOT NULL,
    steam_id TEXT NOT NULL,
    app_id INTEGER NOT NULL DEFAULT 5069970 CHECK(app_id=5069970),
    environment TEXT NOT NULL CHECK(environment IN ('sandbox','live')),
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL CHECK(request_hash ~ '^[0-9a-f]{64}$'),
    pack_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    amount_minor INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    language TEXT NOT NULL CHECK(language IN ('en','fr','es','de','it','pt')),
    currency TEXT NOT NULL DEFAULT 'EUR' CHECK(currency='EUR'),
    state TEXT NOT NULL DEFAULT 'NEW' CHECK(state IN
      ('NEW','INIT_UNKNOWN','AWAITING','FINALIZE_UNKNOWN','PAID','FAILED','CANCELLED','REVOKED','REVIEW')),
    trans_id TEXT CHECK(trans_id ~ '^[1-9][0-9]{0,19}$' AND trans_id::numeric<18446744073709551616),
    UNIQUE(user_id,idempotency_key),
    UNIQUE(environment,trans_id),
    FOREIGN KEY(user_id,profile_uuid,career_id) REFERENCES steam_wallet_checkpoints,
    CHECK((pack_id='tokens25' AND quantity=25 AND amount_minor=99 AND item_id=1)
       OR (pack_id='tokens75' AND quantity=75 AND amount_minor=299 AND item_id=2)
       OR (pack_id='tokens150' AND quantity=150 AND amount_minor=499 AND item_id=3))
);
CREATE UNIQUE INDEX IF NOT EXISTS steam_orders_one_unresolved ON steam_orders(user_id)
    WHERE state IN ('NEW','INIT_UNKNOWN','AWAITING','FINALIZE_UNKNOWN');
CREATE TABLE IF NOT EXISTS steam_receipts (
    receipt_id TEXT PRIMARY KEY,
    receipt_seq BIGSERIAL NOT NULL UNIQUE,
    order_id BIGINT NOT NULL UNIQUE REFERENCES steam_orders(order_id),
    state TEXT NOT NULL CHECK(state IN ('confirmed','revoked','manual'))
);
CREATE TABLE IF NOT EXISTS steam_reconciliation (
    environment TEXT NOT NULL CHECK(environment IN ('sandbox','live')),
    window_start TEXT NOT NULL,
    event_count INTEGER NOT NULL CHECK(event_count>=0),
    next_window TEXT NOT NULL,
    complete BOOLEAN NOT NULL,
    PRIMARY KEY(environment,window_start)
);
CREATE OR REPLACE FUNCTION steam_order_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-'state'-'trans_id') IS DISTINCT FROM (to_jsonb(OLD)-'state'-'trans_id')
       OR (OLD.trans_id IS NOT NULL AND NEW.trans_id IS DISTINCT FROM OLD.trans_id) THEN
        RAISE EXCEPTION 'IMMUTABLE_STEAM_ORDER' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER steam_order_immutable BEFORE UPDATE ON steam_orders
    FOR EACH ROW EXECUTE FUNCTION steam_order_immutable();
CREATE OR REPLACE FUNCTION steam_receipt_protected() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='DELETE' THEN RAISE EXCEPTION 'RECEIPT_DELETE_FORBIDDEN' USING ERRCODE='23514'; END IF;
    IF TG_OP='INSERT' AND NOT EXISTS(SELECT 1 FROM steam_orders WHERE order_id=NEW.order_id AND state='PAID' AND trans_id IS NOT NULL) THEN
        RAISE EXCEPTION 'PAYMENT_NOT_CONFIRMED' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' AND ((to_jsonb(NEW)-'state') IS DISTINCT FROM (to_jsonb(OLD)-'state')
       OR (OLD.state IN ('revoked','manual') AND NEW.state='confirmed')) THEN
        RAISE EXCEPTION 'IMMUTABLE_STEAM_RECEIPT' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER steam_receipt_protected BEFORE INSERT OR UPDATE OR DELETE ON steam_receipts
    FOR EACH ROW EXECUTE FUNCTION steam_receipt_protected();
-- END TABLE DDL
-- Apply only AFTER A5 deliveries exist and their historical references are verified.
DO $$
DECLARE existing pg_constraint%ROWTYPE;
BEGIN
    SELECT * INTO existing FROM pg_constraint
      WHERE conrelid='steam_wallet_deliveries'::regclass AND conname='steam_wallet_a4_receipt_fk';
    IF NOT FOUND THEN
        ALTER TABLE steam_wallet_deliveries ADD CONSTRAINT steam_wallet_a4_receipt_fk
            FOREIGN KEY(receipt_id) REFERENCES steam_receipts(receipt_id);
    ELSIF existing.contype <> 'f' OR existing.confrelid <> 'steam_receipts'::regclass
       OR existing.conkey <> ARRAY[(SELECT attnum FROM pg_attribute WHERE attrelid='steam_wallet_deliveries'::regclass AND attname='receipt_id')]::smallint[]
       OR existing.confkey <> ARRAY[(SELECT attnum FROM pg_attribute WHERE attrelid='steam_receipts'::regclass AND attname='receipt_id')]::smallint[]
       OR existing.confdeltype <> 'a' OR existing.confupdtype <> 'a'
       OR existing.condeferrable OR existing.confmatchtype <> 's' THEN
        RAISE EXCEPTION 'INCOMPATIBLE_STEAM_RECEIPT_FOREIGN_KEY';
    END IF;
    ALTER TABLE steam_wallet_deliveries VALIDATE CONSTRAINT steam_wallet_a4_receipt_fk;
END $$;
COMMIT;
