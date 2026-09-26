"""Dormant internal economy primitives. No HTTP routes or startup/schema writes.

Only trusted server code may supply user_id and economic inputs. A future caller
must authenticate/authorize its business event; this module is not a client API.
Explicit migrations 002/003 are required. Legacy balances are never imported.
"""
import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

_ENABLED = os.environ.get("ENABLE_MODERN_ECONOMY", "") == "true"
_MAX_INT = 9223372036854775807
_MAX_AMOUNT = 1000000000000


class EconomyError(Exception):
    """Stable internal error code; no SQL, credentials or user data exposed."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _gate():
    if not _ENABLED:
        raise EconomyError("ECONOMY_DISABLED")


def _user(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise EconomyError("BAD_USER_ID")
    return value


def _uuid(value, code):
    try:
        if not isinstance(value, str):
            raise ValueError()
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise EconomyError(code) from None


@contextmanager
def _transaction():
    url = (os.environ.get("DATABASE_URL", "") or "").strip()
    if not url:
        raise EconomyError("MODERN_STORAGE_UNAVAILABLE")
    if url.startswith(("postgres://", "postgresql://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    engine = None
    try:
        engine = create_engine(url, future=True, connect_args={"sslmode": "require"})
        with engine.begin() as conn:
            yield conn
    except SQLAlchemyError as exc:
        code = getattr(getattr(exc, "orig", None), "sqlstate", None)
        detail = "MODERN_SCHEMA_NOT_READY" if code in {"42P01", "42703"} else "MODERN_STORAGE_UNAVAILABLE"
        raise EconomyError(detail) from None
    finally:
        if engine is not None:
            engine.dispose()


def read_wallet(user_id):
    """Absent wallet reads as zero; never creates a row."""
    _gate()
    with _transaction() as conn:
        row = conn.execute(text(
            "SELECT balance, version FROM public.modern_wallets WHERE user_id=:user_id"
        ), {"user_id": _user(user_id)}).mappings().first()
        return {"balance": row["balance"] if row else 0,
                "wallet_version": row["version"] if row else 0}


def _receipt(row):
    return {"operation_id": str(row["operation_id"]), "balance": row["balance_after"],
            "wallet_version": row["wallet_version"], "status": "COMMITTED",
            "amount": row["amount"], "source": row["source"]}


def credit(*, operation_id, user_id, amount, source, source_ref=None,
           profile_uuid=None, career_id=None, career_generation=None,
           reverses_operation_id=None):
    return _apply(1, operation_id, user_id, amount, source, source_ref,
                  profile_uuid, career_id, career_generation, reverses_operation_id)


def debit(*, operation_id, user_id, amount, source, source_ref=None,
          profile_uuid=None, career_id=None, career_generation=None,
          reverses_operation_id=None):
    return _apply(-1, operation_id, user_id, amount, source, source_ref,
                  profile_uuid, career_id, career_generation, reverses_operation_id)


def _apply(direction, operation_id, user_id, amount, source, source_ref,
           profile_uuid, career_id, career_generation, reverses_operation_id):
    _gate()
    user_id = _user(user_id)
    operation_id = _uuid(operation_id, "BAD_OPERATION_ID")
    if type(amount) is not int or not 1 <= amount <= _MAX_AMOUNT:
        raise EconomyError("BAD_AMOUNT")
    if not isinstance(source, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", source):
        raise EconomyError("BAD_SOURCE")
    if source_ref is not None and (not isinstance(source_ref, str) or
            not 1 <= len(source_ref) <= 200 or source_ref != source_ref.strip()):
        raise EconomyError("BAD_SOURCE")
    if profile_uuid is not None:
        profile_uuid = _uuid(profile_uuid, "BAD_PROFILE_ID")
    if career_id is not None:
        career_id = _uuid(career_id, "BAD_CAREER_ID")
        if profile_uuid is None or type(career_generation) is not int or not 1 <= career_generation <= _MAX_INT:
            raise EconomyError("BAD_CAREER_CONTEXT")
    elif career_generation is not None:
        raise EconomyError("BAD_CAREER_CONTEXT")
    if reverses_operation_id is not None:
        reverses_operation_id = _uuid(reverses_operation_id, "BAD_OPERATION_ID")
        if reverses_operation_id == operation_id:
            raise EconomyError("BAD_REVERSAL")
    payload = {"contract_version": 1, "user_id": user_id, "amount": direction * amount,
               "source": source, "source_ref": source_ref, "profile_uuid": profile_uuid,
               "career_id": career_id, "career_generation": career_generation,
               "reverses_operation_id": reverses_operation_id}
    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode("utf-8")).hexdigest()
    params = {**payload, "operation_id": operation_id, "request_hash": request_hash}
    with _transaction() as conn:
        # Global operation identity first; collisions only serialize, never alias.
        # Order for new mutations: operation -> career (optional) -> account wallet.
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:operation_id, 504))"), params)
        existing = conn.execute(text(
            "SELECT * FROM public.modern_economic_operations WHERE operation_id=:operation_id"
        ), params).mappings().first()
        if existing is not None:
            if existing["user_id"] != user_id or existing["request_hash"] != request_hash:
                raise EconomyError("IDEMPOTENCY_CONFLICT")
            # Historical receipt remains valid after later operations or career deletion.
            return _receipt(existing)
        if career_id is not None:
            career = conn.execute(text("""
                SELECT state, generation FROM public.modern_careers
                WHERE career_id=:career_id AND user_id=:user_id AND profile_uuid=:profile_uuid
                FOR SHARE
            """), params).mappings().first()
            if career is None:
                raise EconomyError("CAREER_NOT_FOUND")
            if career["generation"] != career_generation:
                raise EconomyError("GENERATION_MISMATCH")
            if career["state"] != "ACTIVE":
                raise EconomyError("CAREER_DELETED")
        conn.execute(text("""
            INSERT INTO public.modern_wallets (user_id) VALUES (:user_id)
            ON CONFLICT (user_id) DO NOTHING
        """), params)
        wallet = conn.execute(text("""
            SELECT balance, version FROM public.modern_wallets WHERE user_id=:user_id FOR UPDATE
        """), params).mappings().one()
        if reverses_operation_id is not None:
            original = conn.execute(text("""
                SELECT amount FROM public.modern_economic_operations
                WHERE operation_id=:reverses_operation_id AND user_id=:user_id
            """), params).mappings().first()
            if original is None or original["amount"] != -params["amount"]:
                raise EconomyError("BAD_REVERSAL")
            if conn.execute(text("""
                SELECT 1 FROM public.modern_economic_operations
                WHERE reverses_operation_id=:reverses_operation_id
            """), params).first() is not None:
                raise EconomyError("BAD_REVERSAL")
        balance = wallet["balance"] + params["amount"]
        if balance < 0:
            raise EconomyError("INSUFFICIENT_FUNDS")
        if balance > _MAX_INT or wallet["version"] == _MAX_INT:
            raise EconomyError("WALLET_LIMIT_REACHED")
        params.update(balance_before=wallet["balance"], balance_after=balance,
                      wallet_version=wallet["version"] + 1)
        row = conn.execute(text("""
            INSERT INTO public.modern_economic_operations
              (operation_id, user_id, profile_uuid, career_id, career_generation, amount,
               source, source_ref, request_hash, balance_before, balance_after,
               wallet_version, reverses_operation_id)
            VALUES (:operation_id, :user_id, :profile_uuid, :career_id, :career_generation,
                    :amount, :source, :source_ref, :request_hash, :balance_before,
                    :balance_after, :wallet_version, :reverses_operation_id)
            RETURNING operation_id, balance_after, wallet_version, amount, source
        """), params).mappings().one()
        conn.execute(text("""
            UPDATE public.modern_wallets SET balance=:balance_after, version=:wallet_version,
                updated_at=NOW() WHERE user_id=:user_id
        """), params)
        return _receipt(row)
