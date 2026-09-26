-- Explicit operator migration only; never startup or route execution.
-- Additive; no legacy data transfer and no down migration.
BEGIN;
SELECT pg_advisory_xact_lock(504, 2);
DO $preflight$
DECLARE actual JSONB;
BEGIN
    IF to_regclass('public.modern_wallets') IS NOT NULL THEN
    SELECT jsonb_build_object(
 'relation',(SELECT jsonb_build_array(relkind,relpersistence,relrowsecurity,relforcerowsecurity) FROM pg_class WHERE oid='public.modern_wallets'::regclass),
 'columns',(SELECT jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull,pg_get_expr(d.adbin,d.adrelid),attidentity,attgenerated) ORDER BY attnum) FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE a.attrelid='public.modern_wallets'::regclass AND attnum>0 AND NOT attisdropped),
 'constraints',(SELECT jsonb_object_agg(conname,jsonb_build_array(pg_get_constraintdef(oid),convalidated,condeferrable,condeferred)) FROM pg_constraint WHERE conrelid='public.modern_wallets'::regclass AND contype<>'n'),
 'indexes',(SELECT jsonb_object_agg(c.relname,jsonb_build_array(pg_get_indexdef(i.indexrelid),i.indisvalid,i.indisready)) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid WHERE indrelid='public.modern_wallets'::regclass),
 'triggers',(SELECT jsonb_object_agg(tgname,jsonb_build_array(pg_get_triggerdef(oid),tgenabled)) FROM pg_trigger WHERE tgrelid='public.modern_wallets'::regclass AND NOT tgisinternal),
 'rules',(SELECT jsonb_object_agg(rulename,pg_get_ruledef(oid)) FROM pg_rewrite WHERE ev_class='public.modern_wallets'::regclass)
) INTO actual;
    IF actual IS DISTINCT FROM $expected${"rules": null, "columns": [["user_id", "text", true, null, "", ""], ["balance", "bigint", true, "0", "", ""], ["version", "bigint", true, "0", "", ""], ["created_at", "timestamp with time zone", true, "now()", "", ""], ["updated_at", "timestamp with time zone", true, "now()", "", ""]], "indexes": {"modern_wallets_pkey": ["CREATE UNIQUE INDEX modern_wallets_pkey ON public.modern_wallets USING btree (user_id)", true, true]}, "relation": ["r", "p", false, false], "triggers": null, "constraints": {"modern_wallets_pkey": ["PRIMARY KEY (user_id)", true, false, false], "modern_wallets_balance_check": ["CHECK ((balance >= 0))", true, false, false], "modern_wallets_user_id_check": ["CHECK (((user_id <> ''::text) AND (user_id = btrim(user_id))))", true, false, false], "modern_wallets_version_check": ["CHECK ((version >= 0))", true, false, false]}}$expected$::jsonb THEN
        RAISE EXCEPTION 'P04_002_SCHEMA_MISMATCH';
    END IF;
    END IF;
END
$preflight$;
CREATE TABLE IF NOT EXISTS public.modern_wallets (
    user_id TEXT PRIMARY KEY CHECK (user_id <> '' AND user_id = btrim(user_id)),
    balance BIGINT NOT NULL DEFAULT 0 CHECK (balance >= 0),
    version BIGINT NOT NULL DEFAULT 0 CHECK (version >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fail closed: replay does not repair incompatible existing objects.
DO $verify$
DECLARE actual JSONB;
BEGIN
    SELECT jsonb_build_object(
 'relation',(SELECT jsonb_build_array(relkind,relpersistence,relrowsecurity,relforcerowsecurity) FROM pg_class WHERE oid='public.modern_wallets'::regclass),
 'columns',(SELECT jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull,pg_get_expr(d.adbin,d.adrelid),attidentity,attgenerated) ORDER BY attnum) FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE a.attrelid='public.modern_wallets'::regclass AND attnum>0 AND NOT attisdropped),
 'constraints',(SELECT jsonb_object_agg(conname,jsonb_build_array(pg_get_constraintdef(oid),convalidated,condeferrable,condeferred)) FROM pg_constraint WHERE conrelid='public.modern_wallets'::regclass AND contype<>'n'),
 'indexes',(SELECT jsonb_object_agg(c.relname,jsonb_build_array(pg_get_indexdef(i.indexrelid),i.indisvalid,i.indisready)) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid WHERE indrelid='public.modern_wallets'::regclass),
 'triggers',(SELECT jsonb_object_agg(tgname,jsonb_build_array(pg_get_triggerdef(oid),tgenabled)) FROM pg_trigger WHERE tgrelid='public.modern_wallets'::regclass AND NOT tgisinternal),
 'rules',(SELECT jsonb_object_agg(rulename,pg_get_ruledef(oid)) FROM pg_rewrite WHERE ev_class='public.modern_wallets'::regclass)
) INTO actual;
    IF actual IS DISTINCT FROM $expected${"rules": null, "columns": [["user_id", "text", true, null, "", ""], ["balance", "bigint", true, "0", "", ""], ["version", "bigint", true, "0", "", ""], ["created_at", "timestamp with time zone", true, "now()", "", ""], ["updated_at", "timestamp with time zone", true, "now()", "", ""]], "indexes": {"modern_wallets_pkey": ["CREATE UNIQUE INDEX modern_wallets_pkey ON public.modern_wallets USING btree (user_id)", true, true]}, "relation": ["r", "p", false, false], "triggers": null, "constraints": {"modern_wallets_pkey": ["PRIMARY KEY (user_id)", true, false, false], "modern_wallets_balance_check": ["CHECK ((balance >= 0))", true, false, false], "modern_wallets_user_id_check": ["CHECK (((user_id <> ''::text) AND (user_id = btrim(user_id))))", true, false, false], "modern_wallets_version_check": ["CHECK ((version >= 0))", true, false, false]}}$expected$::jsonb THEN
        RAISE EXCEPTION 'P04_002_SCHEMA_MISMATCH';
    END IF;
END
$verify$;
COMMIT;
