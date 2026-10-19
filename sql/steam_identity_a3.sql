-- Explicit migration only. Prerequisite: users(user_id TEXT PRIMARY KEY/UNIQUE).
-- Execute as migration owner in the intended schema, before A5 then A4.
-- Existing associations and permissions are preserved. No backend startup DDL.
-- A divergent relation aborts the entire transaction; repair requires review.
BEGIN;
CREATE TABLE IF NOT EXISTS steam_identity_links (
  steam_id TEXT PRIMARY KEY CHECK (steam_id ~ '^[1-9][0-9]{0,19}$'),
  user_id TEXT NOT NULL UNIQUE REFERENCES users(user_id),
  app_id INTEGER NOT NULL CHECK (app_id = 5069970),
  service_identity TEXT NOT NULL CHECK (service_identity = 'basketcorp-identity-v1'),
  linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
DO $$
DECLARE
  target regclass := to_regclass(format('%I.steam_identity_links', current_schema()));
  users_table regclass := to_regclass('users');
  user_column smallint;
  actual_columns jsonb;
  expected_columns constant jsonb := '[
    [1,"steam_id","text",true,"","",null],
    [2,"user_id","text",true,"","",null],
    [3,"app_id","integer",true,"","",null],
    [4,"service_identity","text",true,"","",null],
    [5,"linked_at","timestamp with time zone",true,"","","now()"]
  ]'::jsonb;
  checks text[];
BEGIN
  IF target IS NULL OR users_table IS NULL OR NOT EXISTS (
    SELECT 1 FROM pg_class WHERE oid=target AND relkind='r' AND relpersistence='p'
      AND NOT relispartition AND NOT relrowsecurity AND NOT relforcerowsecurity
  ) THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: relation kind or security';
  END IF;
  EXECUTE format('LOCK TABLE %s IN SHARE MODE', target);
  SELECT attnum INTO user_column FROM pg_attribute
    WHERE attrelid=users_table AND attname='user_id' AND NOT attisdropped
      AND atttypid='text'::regtype;
  IF user_column IS NULL THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: users prerequisite';
  END IF;
  SELECT jsonb_agg(jsonb_build_array(a.attnum,a.attname,format_type(a.atttypid,a.atttypmod),
      a.attnotnull,a.attidentity,a.attgenerated,pg_get_expr(d.adbin,d.adrelid)) ORDER BY a.attnum)
    INTO actual_columns
    FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
    WHERE a.attrelid=target AND a.attnum>0 AND NOT a.attisdropped;
  IF actual_columns IS DISTINCT FROM expected_columns OR EXISTS (
    SELECT 1 FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid
      WHERE a.attrelid=target AND a.attnum>0 AND (a.attisdropped OR a.attcollation<>t.typcollation)
  ) THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: columns, defaults or collation';
  END IF;
  -- PostgreSQL 18 also catalogs NOT NULL constraints (type n); columns above
  -- validate them. All other constraints must match, irrespective of names.
  IF (SELECT count(*) FROM pg_constraint WHERE conrelid=target AND contype<>'n')<>6
     OR EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid=target
                AND (NOT convalidated OR condeferrable OR condeferred OR NOT conenforced))
     OR NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid=target AND contype='p' AND conkey=ARRAY[1]::smallint[])
     OR NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid=target AND contype='u' AND conkey=ARRAY[2]::smallint[])
     OR NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid=target AND contype='f'
        AND conkey=ARRAY[2]::smallint[] AND confrelid=users_table AND confkey=ARRAY[user_column]
        AND confupdtype='a' AND confdeltype='a' AND confmatchtype='s') THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: keys or references';
  END IF;
  SELECT array_agg(pg_get_expr(conbin,conrelid) ORDER BY pg_get_expr(conbin,conrelid))
    INTO checks FROM pg_constraint WHERE conrelid=target AND contype='c';
  IF checks IS DISTINCT FROM ARRAY[
      '(app_id = 5069970)',
      '(service_identity = ''basketcorp-identity-v1''::text)',
      '(steam_id ~ ''^[1-9][0-9]{0,19}$''::text)'
    ]::text[] THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: Steam checks';
  END IF;
  -- FK enforcement uses internal triggers on both this table and users.
  -- Catalog flags alone do not detect DISABLE TRIGGER on either side.
  IF EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_constraint c ON c.oid=t.tgconstraint
      WHERE c.conrelid=target AND c.contype='f' AND t.tgenabled NOT IN ('O','A'))
     OR (SELECT count(*) FROM pg_trigger t JOIN pg_constraint c ON c.oid=t.tgconstraint
      WHERE c.conrelid=target AND c.contype='f' AND t.tgisinternal)<4 THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: foreign key enforcement';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_inherits WHERE inhrelid=target OR inhparent=target)
     OR EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid=target AND NOT tgisinternal)
     OR EXISTS (SELECT 1 FROM pg_rewrite WHERE ev_class=target)
     OR EXISTS (SELECT 1 FROM pg_index i WHERE i.indrelid=target AND
       (NOT i.indisvalid OR NOT i.indisready OR i.indisexclusion OR
        (i.indisunique AND NOT EXISTS (SELECT 1 FROM pg_constraint c
          WHERE c.conrelid=target AND c.conindid=i.indexrelid AND c.contype IN ('p','u'))))) THEN
    RAISE EXCEPTION 'A3_SCHEMA_DIVERGENCE: inherited or additional behavior';
  END IF;
END $$;
COMMIT;
