-- Explicit operator migration only; never startup or route execution.
-- Additive; no legacy data transfer and no down migration.
BEGIN;
SELECT pg_advisory_xact_lock(504, 3);
DO $preflight$
DECLARE actual JSONB;
BEGIN
    IF to_regclass('public.modern_economic_operations') IS NOT NULL THEN
    SELECT jsonb_build_object(
 'relation',(SELECT jsonb_build_array(relkind,relpersistence,relrowsecurity,relforcerowsecurity) FROM pg_class WHERE oid='public.modern_economic_operations'::regclass),
 'columns',(SELECT jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull,pg_get_expr(d.adbin,d.adrelid),attidentity,attgenerated) ORDER BY attnum) FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE a.attrelid='public.modern_economic_operations'::regclass AND attnum>0 AND NOT attisdropped),
 'constraints',(SELECT jsonb_object_agg(conname,jsonb_build_array(pg_get_constraintdef(oid),convalidated,condeferrable,condeferred)) FROM pg_constraint WHERE conrelid='public.modern_economic_operations'::regclass AND contype<>'n'),
 'indexes',(SELECT jsonb_object_agg(c.relname,jsonb_build_array(pg_get_indexdef(i.indexrelid),i.indisvalid,i.indisready)) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid WHERE indrelid='public.modern_economic_operations'::regclass),
 'triggers',(SELECT jsonb_object_agg(tgname,jsonb_build_array(pg_get_triggerdef(oid),tgenabled)) FROM pg_trigger WHERE tgrelid='public.modern_economic_operations'::regclass AND NOT tgisinternal),
 'rules',(SELECT jsonb_object_agg(rulename,pg_get_ruledef(oid)) FROM pg_rewrite WHERE ev_class='public.modern_economic_operations'::regclass)
) INTO actual;
    IF actual IS DISTINCT FROM $expected${"rules": null, "columns": [["operation_id", "uuid", true, null, "", ""], ["user_id", "text", true, null, "", ""], ["profile_uuid", "uuid", false, null, "", ""], ["career_id", "uuid", false, null, "", ""], ["career_generation", "bigint", false, null, "", ""], ["amount", "bigint", true, null, "", ""], ["source", "text", true, null, "", ""], ["source_ref", "text", false, null, "", ""], ["request_hash", "text", true, null, "", ""], ["balance_before", "bigint", true, null, "", ""], ["balance_after", "bigint", true, null, "", ""], ["wallet_version", "bigint", true, null, "", ""], ["reverses_operation_id", "uuid", false, null, "", ""], ["committed_at", "timestamp with time zone", true, "now()", "", ""]], "indexes": {"modern_economic_operations_pkey": ["CREATE UNIQUE INDEX modern_economic_operations_pkey ON public.modern_economic_operations USING btree (operation_id)", true, true], "modern_economic_user_version_uq": ["CREATE UNIQUE INDEX modern_economic_user_version_uq ON public.modern_economic_operations USING btree (user_id, wallet_version)", true, true], "modern_economic_operations_reverses_operation_id_key": ["CREATE UNIQUE INDEX modern_economic_operations_reverses_operation_id_key ON public.modern_economic_operations USING btree (reverses_operation_id)", true, true]}, "relation": ["r", "p", false, false], "triggers": {"modern_economic_immutable": ["CREATE TRIGGER modern_economic_immutable BEFORE DELETE OR UPDATE OR TRUNCATE ON public.modern_economic_operations FOR EACH STATEMENT EXECUTE FUNCTION modern_economic_reject_mutation()", "O"]}, "constraints": {"modern_economic_balance_ck": ["CHECK (((balance_after)::numeric = ((balance_before)::numeric + (amount)::numeric)))", true, false, false], "modern_economic_context_ck": ["CHECK ((((career_id IS NULL) AND (career_generation IS NULL)) OR ((career_id IS NOT NULL) AND (profile_uuid IS NOT NULL) AND (career_generation IS NOT NULL) AND (career_generation >= 1))))", true, false, false], "modern_economic_reversal_ck": ["CHECK (((reverses_operation_id IS NULL) OR (reverses_operation_id <> operation_id)))", true, false, false], "modern_economic_operations_pkey": ["PRIMARY KEY (operation_id)", true, false, false], "modern_economic_user_version_uq": ["UNIQUE (user_id, wallet_version)", true, false, false], "modern_economic_operations_amount_check": ["CHECK (((amount <> 0) AND ((amount >= '-1000000000000'::bigint) AND (amount <= '1000000000000'::bigint))))", true, false, false], "modern_economic_operations_source_check": ["CHECK ((source ~ '^[A-Z][A-Z0-9_]{0,63}$'::text))", true, false, false], "modern_economic_operations_user_id_fkey": ["FOREIGN KEY (user_id) REFERENCES modern_wallets(user_id)", true, false, false], "modern_economic_operations_source_ref_check": ["CHECK (((source_ref IS NULL) OR (((length(source_ref) >= 1) AND (length(source_ref) <= 200)) AND (source_ref = btrim(source_ref)))))", true, false, false], "modern_economic_operations_request_hash_check": ["CHECK ((request_hash ~ '^[0-9a-f]{64}$'::text))", true, false, false], "modern_economic_operations_balance_after_check": ["CHECK ((balance_after >= 0))", true, false, false], "modern_economic_operations_balance_before_check": ["CHECK ((balance_before >= 0))", true, false, false], "modern_economic_operations_wallet_version_check": ["CHECK ((wallet_version >= 1))", true, false, false], "modern_economic_operations_reverses_operation_id_key": ["UNIQUE (reverses_operation_id)", true, false, false], "modern_economic_operations_reverses_operation_id_fkey": ["FOREIGN KEY (reverses_operation_id) REFERENCES modern_economic_operations(operation_id)", true, false, false]}}$expected$::jsonb THEN
        RAISE EXCEPTION 'P04_003_SCHEMA_MISMATCH';
    END IF;
    SELECT jsonb_build_array(prosrc,prorettype::regtype::text,(SELECT lanname FROM pg_language WHERE oid=prolang),prosecdef,provolatile,proconfig,pronargs) FROM pg_proc WHERE oid='public.modern_economic_reject_mutation()'::regprocedure INTO actual;
    IF actual IS DISTINCT FROM $expected$["\nBEGIN\n    RAISE EXCEPTION 'MODERN_LEDGER_IMMUTABLE' USING ERRCODE = '55000';\nEND\n", "trigger", "plpgsql", false, "v", null, 0]$expected$::jsonb THEN
        RAISE EXCEPTION 'P04_003_TRIGGER_MISMATCH';
    END IF;
    END IF;
END
$preflight$;
CREATE TABLE IF NOT EXISTS public.modern_economic_operations (
    operation_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES public.modern_wallets(user_id),
    profile_uuid UUID NULL,
    career_id UUID NULL,
    career_generation BIGINT NULL,
    amount BIGINT NOT NULL CHECK (amount <> 0 AND amount BETWEEN -1000000000000 AND 1000000000000),
    source TEXT NOT NULL CHECK (source ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    source_ref TEXT NULL CHECK (source_ref IS NULL OR (length(source_ref) BETWEEN 1 AND 200 AND source_ref = btrim(source_ref))),
    request_hash TEXT NOT NULL CHECK (request_hash ~ '^[0-9a-f]{64}$'),
    balance_before BIGINT NOT NULL CHECK (balance_before >= 0),
    balance_after BIGINT NOT NULL CHECK (balance_after >= 0),
    wallet_version BIGINT NOT NULL CHECK (wallet_version >= 1),
    reverses_operation_id UUID NULL UNIQUE REFERENCES public.modern_economic_operations(operation_id),
    committed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT modern_economic_context_ck CHECK (
        (career_id IS NULL AND career_generation IS NULL)
        OR (career_id IS NOT NULL AND profile_uuid IS NOT NULL AND career_generation IS NOT NULL AND career_generation >= 1)
    ),
    CONSTRAINT modern_economic_balance_ck CHECK (balance_after::numeric = balance_before::numeric + amount::numeric),
    CONSTRAINT modern_economic_reversal_ck CHECK (reverses_operation_id IS NULL OR reverses_operation_id <> operation_id),
    CONSTRAINT modern_economic_user_version_uq UNIQUE (user_id, wallet_version)
);
-- One immutable receipt per operation. source_ref is provenance, NOT a unique key.
-- Reversal links allow one full compensation per original operation.
DO $install$
BEGIN
    IF to_regprocedure('public.modern_economic_reject_mutation()') IS NULL THEN
        EXECUTE $ddl$CREATE FUNCTION public.modern_economic_reject_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $body$
BEGIN
    RAISE EXCEPTION 'MODERN_LEDGER_IMMUTABLE' USING ERRCODE = '55000';
END
$body$$ddl$;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='public.modern_economic_operations'::regclass AND tgname='modern_economic_immutable') THEN
        CREATE TRIGGER modern_economic_immutable
        BEFORE UPDATE OR DELETE OR TRUNCATE ON public.modern_economic_operations
        FOR EACH STATEMENT EXECUTE FUNCTION public.modern_economic_reject_mutation();
    END IF;
END
$install$;

-- Fail closed: replay does not repair incompatible existing objects.
DO $verify$
DECLARE actual JSONB;
BEGIN
    SELECT jsonb_build_object(
 'relation',(SELECT jsonb_build_array(relkind,relpersistence,relrowsecurity,relforcerowsecurity) FROM pg_class WHERE oid='public.modern_economic_operations'::regclass),
 'columns',(SELECT jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull,pg_get_expr(d.adbin,d.adrelid),attidentity,attgenerated) ORDER BY attnum) FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE a.attrelid='public.modern_economic_operations'::regclass AND attnum>0 AND NOT attisdropped),
 'constraints',(SELECT jsonb_object_agg(conname,jsonb_build_array(pg_get_constraintdef(oid),convalidated,condeferrable,condeferred)) FROM pg_constraint WHERE conrelid='public.modern_economic_operations'::regclass AND contype<>'n'),
 'indexes',(SELECT jsonb_object_agg(c.relname,jsonb_build_array(pg_get_indexdef(i.indexrelid),i.indisvalid,i.indisready)) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid WHERE indrelid='public.modern_economic_operations'::regclass),
 'triggers',(SELECT jsonb_object_agg(tgname,jsonb_build_array(pg_get_triggerdef(oid),tgenabled)) FROM pg_trigger WHERE tgrelid='public.modern_economic_operations'::regclass AND NOT tgisinternal),
 'rules',(SELECT jsonb_object_agg(rulename,pg_get_ruledef(oid)) FROM pg_rewrite WHERE ev_class='public.modern_economic_operations'::regclass)
) INTO actual;
    IF actual IS DISTINCT FROM $expected${"rules": null, "columns": [["operation_id", "uuid", true, null, "", ""], ["user_id", "text", true, null, "", ""], ["profile_uuid", "uuid", false, null, "", ""], ["career_id", "uuid", false, null, "", ""], ["career_generation", "bigint", false, null, "", ""], ["amount", "bigint", true, null, "", ""], ["source", "text", true, null, "", ""], ["source_ref", "text", false, null, "", ""], ["request_hash", "text", true, null, "", ""], ["balance_before", "bigint", true, null, "", ""], ["balance_after", "bigint", true, null, "", ""], ["wallet_version", "bigint", true, null, "", ""], ["reverses_operation_id", "uuid", false, null, "", ""], ["committed_at", "timestamp with time zone", true, "now()", "", ""]], "indexes": {"modern_economic_operations_pkey": ["CREATE UNIQUE INDEX modern_economic_operations_pkey ON public.modern_economic_operations USING btree (operation_id)", true, true], "modern_economic_user_version_uq": ["CREATE UNIQUE INDEX modern_economic_user_version_uq ON public.modern_economic_operations USING btree (user_id, wallet_version)", true, true], "modern_economic_operations_reverses_operation_id_key": ["CREATE UNIQUE INDEX modern_economic_operations_reverses_operation_id_key ON public.modern_economic_operations USING btree (reverses_operation_id)", true, true]}, "relation": ["r", "p", false, false], "triggers": {"modern_economic_immutable": ["CREATE TRIGGER modern_economic_immutable BEFORE DELETE OR UPDATE OR TRUNCATE ON public.modern_economic_operations FOR EACH STATEMENT EXECUTE FUNCTION modern_economic_reject_mutation()", "O"]}, "constraints": {"modern_economic_balance_ck": ["CHECK (((balance_after)::numeric = ((balance_before)::numeric + (amount)::numeric)))", true, false, false], "modern_economic_context_ck": ["CHECK ((((career_id IS NULL) AND (career_generation IS NULL)) OR ((career_id IS NOT NULL) AND (profile_uuid IS NOT NULL) AND (career_generation IS NOT NULL) AND (career_generation >= 1))))", true, false, false], "modern_economic_reversal_ck": ["CHECK (((reverses_operation_id IS NULL) OR (reverses_operation_id <> operation_id)))", true, false, false], "modern_economic_operations_pkey": ["PRIMARY KEY (operation_id)", true, false, false], "modern_economic_user_version_uq": ["UNIQUE (user_id, wallet_version)", true, false, false], "modern_economic_operations_amount_check": ["CHECK (((amount <> 0) AND ((amount >= '-1000000000000'::bigint) AND (amount <= '1000000000000'::bigint))))", true, false, false], "modern_economic_operations_source_check": ["CHECK ((source ~ '^[A-Z][A-Z0-9_]{0,63}$'::text))", true, false, false], "modern_economic_operations_user_id_fkey": ["FOREIGN KEY (user_id) REFERENCES modern_wallets(user_id)", true, false, false], "modern_economic_operations_source_ref_check": ["CHECK (((source_ref IS NULL) OR (((length(source_ref) >= 1) AND (length(source_ref) <= 200)) AND (source_ref = btrim(source_ref)))))", true, false, false], "modern_economic_operations_request_hash_check": ["CHECK ((request_hash ~ '^[0-9a-f]{64}$'::text))", true, false, false], "modern_economic_operations_balance_after_check": ["CHECK ((balance_after >= 0))", true, false, false], "modern_economic_operations_balance_before_check": ["CHECK ((balance_before >= 0))", true, false, false], "modern_economic_operations_wallet_version_check": ["CHECK ((wallet_version >= 1))", true, false, false], "modern_economic_operations_reverses_operation_id_key": ["UNIQUE (reverses_operation_id)", true, false, false], "modern_economic_operations_reverses_operation_id_fkey": ["FOREIGN KEY (reverses_operation_id) REFERENCES modern_economic_operations(operation_id)", true, false, false]}}$expected$::jsonb THEN
        RAISE EXCEPTION 'P04_003_SCHEMA_MISMATCH';
    END IF;
    SELECT jsonb_build_array(prosrc,prorettype::regtype::text,(SELECT lanname FROM pg_language WHERE oid=prolang),prosecdef,provolatile,proconfig,pronargs) FROM pg_proc WHERE oid='public.modern_economic_reject_mutation()'::regprocedure INTO actual;
    IF actual IS DISTINCT FROM $expected$["\nBEGIN\n    RAISE EXCEPTION 'MODERN_LEDGER_IMMUTABLE' USING ERRCODE = '55000';\nEND\n", "trigger", "plpgsql", false, "v", null, 0]$expected$::jsonb THEN
        RAISE EXCEPTION 'P04_003_TRIGGER_MISMATCH';
    END IF;
END
$verify$;
COMMIT;
