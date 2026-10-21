-- Explicit installation only. Never run at startup. No historical backfill.
BEGIN;
CREATE TABLE apple_purchase_intents_v1 (
 intent_id UUID PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(user_id),
 request_id UUID NOT NULL,
 product_id TEXT NOT NULL CHECK (product_id IN ('com.basketcorp.tokens25','com.basketcorp.tokens75','com.basketcorp.tokens150')),
 transaction_id TEXT UNIQUE,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 UNIQUE(user_id,request_id)
);
-- Retain the SAME legacy global transaction uniqueness; abort if unavailable.
DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class n ON n.oid=i.indexrelid
 WHERE i.indrelid='public.payments'::regclass AND n.relname='payments_apple_transaction_id_uq'
 AND i.indisunique AND i.indisvalid AND i.indisready
 AND pg_get_indexdef(i.indexrelid) = 'CREATE UNIQUE INDEX payments_apple_transaction_id_uq ON public.payments USING btree (apple_transaction_id) WHERE (apple_transaction_id IS NOT NULL)')
 THEN RAISE EXCEPTION 'Apple global uniqueness required'; END IF;
END $$;
COMMIT;
