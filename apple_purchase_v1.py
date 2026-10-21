"""Opt-in V1 purchase correlation. No career registry, cloud or startup SQL."""
import os
import uuid
import hashlib
import hmac
import re
from fastapi import HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError


def reject(code, status=409):
    raise HTTPException(status_code=status, detail=code)


def canonical_uuid(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        reject('APPLE_INTENT_BAD_UUID', 400)


class Purchases:
    def __init__(self, engine, verifier, products, upsert, credit):
        self.engine, self.verifier, self.products = engine, verifier, products
        self.upsert, self.credit = upsert, credit

    def intent(self, user, request_id, product):
        request_id = canonical_uuid(request_id)
        if product not in self.products:
            reject('UNKNOWN_APPLE_PRODUCT', 400)
        token = str(uuid.uuid4())
        while token.replace('-', '') == user.replace('-', ''):
            token = str(uuid.uuid4())
        with self.engine().begin() as c:
            c.execute(text('''INSERT INTO apple_purchase_intents_v1
                (intent_id,user_id,request_id,product_id) VALUES (:i,:u,:r,:p)
                ON CONFLICT (user_id,request_id) DO NOTHING'''),dict(i=token,u=user,r=request_id,p=product))
            row = c.execute(text('''SELECT intent_id,product_id FROM apple_purchase_intents_v1
                WHERE user_id=:u AND request_id=:r FOR UPDATE'''),dict(u=user,r=request_id)).first()
            if row is None or row[1] != product:
                reject('APPLE_INTENT_CONFLICT')
            return dict(intent_id=str(row[0]), product_id=product)

    def verify(self, user, signed):
        tx, txid, product, pack, price, currency = self.verifier(signed)
        token = canonical_uuid(getattr(tx, 'appAccountToken', None))
        if token.replace('-', '') == user.replace('-', ''):
            reject('APPLE_LEGACY_TRANSACTION')
        if product not in self.products or pack != self.products[product]:
            reject('APPLE_PRODUCT_MISMATCH')
        with self.engine().begin() as c:
            row = c.execute(text('''SELECT user_id,product_id,transaction_id FROM apple_purchase_intents_v1
                WHERE intent_id=:i FOR UPDATE'''),dict(i=token)).first()
            if row is None or row[0] != user:
                reject('APPLE_INTENT_NOT_OWNED', 403)
            if row[1] != product or (row[2] is not None and row[2] != txid):
                reject('APPLE_INTENT_TRANSACTION_CONFLICT')
            # The legacy global unique index and credit lock remain authoritative.
            self.upsert(c, payment_id=uuid.uuid4().hex, user_id=user, profile_uuid=user,
                        transaction_id=txid, product_id=product, pack=pack,
                        price_milliunits=price, currency=currency)
            credited, _balance = self.credit(c, transaction_id=txid, user_id=user,
                                            profile_uuid=user, tokens=pack['tokens'])
            c.execute(text('''UPDATE apple_purchase_intents_v1 SET transaction_id=:t
                WHERE intent_id=:i'''),dict(t=txid,i=token))
            return dict(ok=True, verified=True, credited=True, credited_now=credited,
                        intent_id=token, transaction_id=txid, product_id=product,
                        tokens=pack['tokens'], pack_id=pack['pack_id'])

    def anonymous_intent(self, secret, request_id, product):
        digest = secret_digest(secret)
        request_id = canonical_uuid(request_id)
        if product not in self.products:
            reject('UNKNOWN_APPLE_PRODUCT', 400)
        with self.engine().begin() as c:
            c.execute(text("""INSERT INTO apple_purchase_intents_v1
                (intent_id,user_id,request_id,product_id,secret_hash)
                VALUES (:i,NULL,:r,:p,:s)
                ON CONFLICT (request_id) WHERE user_id IS NULL DO NOTHING"""),
                dict(i=str(uuid.uuid4()), r=request_id, p=product, s=digest))
            row = c.execute(text("""SELECT intent_id,product_id,secret_hash
                FROM apple_purchase_intents_v1 WHERE user_id IS NULL AND request_id=:r
                FOR UPDATE"""), dict(r=request_id)).first()
            if row is None or not hmac.compare_digest(row[2], digest):
                reject('APPLE_INTENT_NOT_OWNED', 403)
            if row[1] != product:
                reject('APPLE_INTENT_CONFLICT')
            return dict(intent_id=str(row[0]), product_id=product)

    def anonymous_verify(self, secret, intent_id, signed):
        digest = secret_digest(secret)
        intent_id = canonical_uuid(intent_id)
        # Authenticate the capability before expensive Apple signature verification.
        with self.engine().connect() as c:
            row = c.execute(text("""SELECT secret_hash FROM apple_purchase_intents_v1
                WHERE intent_id=:i AND user_id IS NULL"""), dict(i=intent_id)).first()
            if row is None or not hmac.compare_digest(row[0], digest):
                reject('APPLE_INTENT_NOT_OWNED', 403)
        tx, txid, product, pack, price, currency = self.verifier(signed)
        token = canonical_uuid(getattr(tx, 'appAccountToken', None))
        if token != intent_id or not isinstance(txid, str) or not txid:
            reject('APPLE_INTENT_TRANSACTION_CONFLICT')
        if product not in self.products or pack != self.products[product]:
            reject('APPLE_PRODUCT_MISMATCH')
        proof = dict(intent_id=token, transaction_id=txid, product_id=product,
                     tokens=pack['tokens'], pack_id=pack['pack_id'],
                     price_milliunits=price, currency=currency)
        import json
        with self.engine().begin() as c:
            row = c.execute(text("""SELECT secret_hash,product_id,transaction_id,verified_receipt
                FROM apple_purchase_intents_v1 WHERE intent_id=:i AND user_id IS NULL
                FOR UPDATE"""), dict(i=token)).first()
            if row is None or not hmac.compare_digest(row[0], digest):
                reject('APPLE_INTENT_NOT_OWNED', 403)
            if row[1] != product or (row[2] is not None and (row[2] != txid or row[3] != proof)):
                reject('APPLE_INTENT_TRANSACTION_CONFLICT')
            # SQL triggers atomically reserve this transaction across BOTH contracts.
            if row[2] is None:
                c.execute(text("""UPDATE apple_purchase_intents_v1
                    SET transaction_id=:t,verified_receipt=CAST(:r AS JSONB)
                    WHERE intent_id=:i"""),dict(t=txid,r=json.dumps(proof),i=token))
            # No account, payments row or account-wallet credit for anonymous purchases.
            return dict(ok=True, verified=True, delivery_authorized=True, **proof)


def secret_digest(secret):
    if not isinstance(secret, str) or re.fullmatch(r'[0-9a-f]{64}', secret) is None:
        reject('APPLE_INTENT_BAD_SECRET', 400)
    return hashlib.sha256(bytes.fromhex(secret)).hexdigest()


def register(app, authenticate, engine, verifier, products, upsert, credit):
    def service(request):
        if os.environ.get('ENABLE_APPLE_PURCHASE_INTENTS_V1', '') != 'true':
            reject('APPLE_INTENTS_DISABLED', 503)
        user = authenticate(request.headers.get('authorization', '')).get('sub')
        if not isinstance(user, str) or not user:
            reject('INVALID_TOKEN', 401)
        def db():
            result = engine()
            if result is None:
                reject('PAYMENTS_DB_NOT_READY', 503)
            return result
        return user, Purchases(db, verifier, products, upsert, credit)

    async def payload(request, keys):
        try:
            data = await request.json()
        except Exception:
            reject('APPLE_INTENT_BAD_BODY', 400)
        if not isinstance(data, dict) or set(data) != set(keys) or any(not isinstance(data[k], str) or not data[k] for k in keys):
            reject('APPLE_INTENT_BAD_BODY', 400)
        return data

    def run(fn, *args):
        try:
            return fn(*args)
        except SQLAlchemyError:
            reject('APPLE_INTENT_STORAGE_UNAVAILABLE', 503)

    @app.post('/v1/payments/apple/intents')
    async def apple_purchase_intent(request: Request):
        user, svc = service(request)
        body = await payload(request, ('request_id', 'product_id'))
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(run, svc.intent, user, body['request_id'], body['product_id'])

    @app.post('/v1/payments/apple/intent-verify')
    async def apple_purchase_intent_verify(request: Request):
        user, svc = service(request)
        body = await payload(request, ('signed_transaction_info',))
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(run, svc.verify, user, body['signed_transaction_info'])

    def anonymous_service(request):
        if (os.environ.get('ENABLE_APPLE_PURCHASE_INTENTS_V1', '') != 'true' or
                os.environ.get('ENABLE_APPLE_ANONYMOUS_INTENTS_V1', '') != 'true'):
            reject('APPLE_ANONYMOUS_INTENTS_DISABLED', 503)
        secret = request.headers.get('x-apple-intent-secret', '')
        secret_digest(secret)
        def db():
            result = engine()
            if result is None:
                reject('PAYMENTS_DB_NOT_READY', 503)
            return result
        return secret, Purchases(db, verifier, products, upsert, credit)

    def run_anonymous(fn, *args):
        try:
            return fn(*args)
        except IntegrityError:
            reject('APPLE_TRANSACTION_CONFLICT', 409)
        except SQLAlchemyError:
            reject('APPLE_INTENT_STORAGE_UNAVAILABLE', 503)

    @app.post('/v1/payments/apple/anonymous-intents')
    async def apple_anonymous_intent(request: Request):
        secret, svc = anonymous_service(request)
        body = await payload(request, ('request_id', 'product_id'))
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(run_anonymous, svc.anonymous_intent,
                                      secret, body['request_id'], body['product_id'])

    @app.post('/v1/payments/apple/anonymous-intent-verify')
    async def apple_anonymous_verify(request: Request):
        secret, svc = anonymous_service(request)
        body = await payload(request, ('intent_id', 'signed_transaction_info'))
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(run_anonymous, svc.anonymous_verify,
                                      secret, body['intent_id'], body['signed_transaction_info'])
