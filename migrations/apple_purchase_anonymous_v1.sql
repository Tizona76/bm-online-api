-- Local preparation only. Requires apple_purchase_intents_v1.sql.
-- No payments columns, existing rows, or legacy unique index are changed.
BEGIN;
SET LOCAL lock_timeout = '5s';
LOCK TABLE payments, apple_purchase_intents_v1 IN SHARE ROW EXCLUSIVE MODE;

ALTER TABLE apple_purchase_intents_v1 ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE apple_purchase_intents_v1 ADD COLUMN secret_hash TEXT;
ALTER TABLE apple_purchase_intents_v1 ADD COLUMN verified_receipt JSONB;
ALTER TABLE apple_purchase_intents_v1 ADD CONSTRAINT apple_intent_owner_v1 CHECK (
 (user_id IS NOT NULL AND secret_hash IS NULL AND verified_receipt IS NULL) OR
 (user_id IS NULL AND secret_hash ~ '^[0-9a-f]{64}$' AND secret_hash IS NOT NULL
  AND ((transaction_id IS NULL AND verified_receipt IS NULL) OR
       (transaction_id IS NOT NULL AND verified_receipt IS NOT NULL
        AND jsonb_typeof(verified_receipt) = 'object')))
);
CREATE UNIQUE INDEX apple_anonymous_request_v1_uq
 ON apple_purchase_intents_v1(request_id) WHERE user_id IS NULL;

-- One shared uniqueness arbiter, not a second wallet or delivery ledger.
CREATE TABLE apple_transaction_claims_v1 (
 transaction_id TEXT PRIMARY KEY CHECK (transaction_id <> ''),
 owner_kind TEXT NOT NULL CHECK (owner_kind IN ('historical','anonymous')),
 owner_key TEXT NOT NULL,
 CHECK ((owner_kind = 'historical' AND owner_key = '') OR
        (owner_kind = 'anonymous' AND owner_key <> ''))
);
INSERT INTO apple_transaction_claims_v1(transaction_id,owner_kind,owner_key)
 SELECT apple_transaction_id,'historical','' FROM payments
 WHERE apple_transaction_id IS NOT NULL;

CREATE FUNCTION apple_claim_transaction_v1(t TEXT, k TEXT, o TEXT) RETURNS VOID
 LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
DECLARE claimed TEXT;
BEGIN
 INSERT INTO public.apple_transaction_claims_v1(transaction_id,owner_kind,owner_key)
 VALUES(t,k,o)
 ON CONFLICT(transaction_id) DO UPDATE SET transaction_id=EXCLUDED.transaction_id
 WHERE apple_transaction_claims_v1.owner_kind=EXCLUDED.owner_kind
 AND apple_transaction_claims_v1.owner_key=EXCLUDED.owner_key
 RETURNING transaction_id INTO claimed;
 IF claimed IS NULL THEN
  RAISE EXCEPTION USING ERRCODE='23505', MESSAGE='APPLE_TRANSACTION_ALREADY_CLAIMED';
 END IF;
END $$;

CREATE FUNCTION apple_payments_claim_v1() RETURNS TRIGGER
 LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
BEGIN
 IF NEW.apple_transaction_id IS NOT NULL THEN
  PERFORM public.apple_claim_transaction_v1(NEW.apple_transaction_id,'historical','');
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER apple_payments_claim_v1
 BEFORE INSERT OR UPDATE OF apple_transaction_id ON payments
 FOR EACH ROW EXECUTE FUNCTION apple_payments_claim_v1();

CREATE FUNCTION apple_intent_claim_v1() RETURNS TRIGGER
 LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
BEGIN
 IF TG_OP='UPDATE' THEN
  IF ROW(OLD.intent_id,OLD.user_id,OLD.request_id,OLD.product_id,OLD.secret_hash)
     IS DISTINCT FROM ROW(NEW.intent_id,NEW.user_id,NEW.request_id,NEW.product_id,NEW.secret_hash)
     OR (OLD.transaction_id IS NOT NULL AND
         ROW(OLD.transaction_id,OLD.verified_receipt) IS DISTINCT FROM
         ROW(NEW.transaction_id,NEW.verified_receipt)) THEN
   RAISE EXCEPTION USING ERRCODE='23514', MESSAGE='APPLE_INTENT_IMMUTABLE';
  END IF;
 END IF;
 IF NEW.user_id IS NULL AND NEW.transaction_id IS NOT NULL THEN
  PERFORM public.apple_claim_transaction_v1(NEW.transaction_id,'anonymous',NEW.intent_id::text);
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER apple_intent_claim_v1
 BEFORE INSERT OR UPDATE ON apple_purchase_intents_v1
 FOR EACH ROW EXECUTE FUNCTION apple_intent_claim_v1();
COMMIT;
