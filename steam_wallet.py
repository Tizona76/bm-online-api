"""A5 wallet. No startup I/O, DDL or Valve calls.

STEAM_WALLET_ENABLED=1 enables routes independently of A3. Delivery remains
unavailable unless STEAM_PAYMENTS_ENABLED=1 also enables the A4 registry.
Adapter.lock(conn, receipt_id) MUST lock the authoritative A4 row on the same
connection, after the checkpoint lock, and return immutable identity/quantity
plus current state. A4 refunds must obey that lock order too. No adapter can
be selected or populated by HTTP, configuration strings, or environment data.
"""
import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.concurrency import run_in_threadpool

if __package__:
    from .steam_identity import APP_ID, IDENTITY, Rejected, Store as IdentityStore
else:
    from steam_identity import APP_ID, IDENTITY, Rejected, Store as IdentityStore

MAX_BYTES = 2_000_000
MAX_INTEGER = 2**53 - 1


def reject(status, code):
    raise Rejected(status, code)


def normalize(value, depth=0):
    """Stable integer normalization across Godot JSON / Python; reject NaN."""
    if depth > 64:
        reject(400, 'INVALID_SNAPSHOT')
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) in (int, float):
        if not math.isfinite(value) or abs(value) > MAX_INTEGER:
            reject(400, 'INVALID_NUMBER')
        return int(value) if int(value) == value else value
    if isinstance(value, list):
        return [normalize(v, depth + 1) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: normalize(v, depth + 1) for k, v in value.items()}
    reject(400, 'INVALID_SNAPSHOT')


def encoded(value):
    return json.dumps(normalize(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def fields(body, required):
    if not isinstance(body, dict) or set(body) != set(required):
        reject(400, 'INVALID_REQUEST')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        reject(400, 'INVALID_IDENTIFIER')
    return value


def revision(value):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        reject(400, 'INVALID_REVISION')
    return value


def hash_value(value, initial=False):
    if initial and value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        reject(400, 'INVALID_HASH')
    return value


class UnavailableReceipts:
    def lock(self, conn, receipt_id):
        reject(503, 'STEAM_A4_UNAVAILABLE')


class A4Receipts:
    def lock(self, conn, receipt_id):
        if os.environ.get('STEAM_PAYMENTS_ENABLED') != '1':
            reject(503, 'STEAM_A4_UNAVAILABLE')
        if __package__:
            from .steam_payments import ReceiptSource, environment
        else:
            from steam_payments import ReceiptSource, environment
        value = ReceiptSource().lock(conn, receipt_id)
        if value is not None and value['environment'] != environment():
            reject(409, 'RECEIPT_ENVIRONMENT_CONFLICT')
        return value


class Wallet:
    def __init__(self, engine_provider, receipt_source=None, *, steam_id=None):
        self.steam_id = steam_id
        self.identity = IdentityStore(engine_provider)
        self.source = receipt_source if receipt_source is not None else A4Receipts()

    @contextmanager
    def transaction(self):
        engine = self.identity.engine()
        # Production concurrency depends on PostgreSQL row locks, never SQLite.
        if engine.dialect.name != 'postgresql':
            reject(503, 'STEAM_WALLET_POSTGRES_REQUIRED')
        with engine.begin() as conn:
            yield conn

    def authenticate(self, conn, user):
        if self.steam_id is None:
            self.identity.eligible(conn, user, lock=True)
        else:
            self.identity.eligible_steam(conn, user, self.steam_id, lock=True)
        link = conn.execute(text('''SELECT steam_id,app_id,service_identity
            FROM steam_identity_links WHERE user_id=:u'''), {'u': user}).mappings().first()
        if not link or link['app_id'] != APP_ID or link['service_identity'] != IDENTITY:
            reject(409, 'STEAM_LINK_REQUIRED')
        return link['steam_id']

    def checkpoint(self, conn, args):
        return conn.execute(text('''SELECT * FROM steam_wallet_checkpoints
            WHERE user_id=:u AND profile_uuid=:p AND career_id=:c FOR UPDATE'''), args).mappings().first()

    def args(self, user, body):
        return {'u': user, 'p': identifier(body['profile_uuid']), 'c': identifier(body['career_id'])}

    def snapshot(self, blob, args, metadata=None):
        blob = normalize(copy.deepcopy(blob))
        if not isinstance(blob, dict) or not isinstance(blob.get('wallet'), dict):
            reject(400, 'INVALID_SNAPSHOT')
        tokens = blob['wallet'].get('tokens')
        if type(tokens) is not int or not 0 <= tokens <= MAX_INTEGER:
            reject(400, 'INVALID_BALANCE')
        if 'career_id' in blob and blob['career_id'] != args['c']:
            reject(409, 'DESTINATION_CONFLICT')
        expected = metadata or {'binding': {'user_id': args['u'], 'profile_uuid': args['p'],
                                          'career_id': args['c']}, 'applied_receipts': {}}
        if 'steam_wallet' in blob and blob['steam_wallet'] != expected:
            reject(409, 'RECEIPT_MARKERS_CONFLICT')
        if metadata is not None and 'steam_wallet' not in blob:
            reject(409, 'RECEIPT_MARKERS_CONFLICT')
        blob['steam_wallet'] = copy.deepcopy(expected)
        if len(encoded(blob).encode()) > MAX_BYTES:
            reject(413, 'SNAPSHOT_TOO_LARGE')
        return blob

    def parent(self, row, body):
        if not row or row['revision'] != body['expected_revision']:
            reject(409, 'REVISION_CONFLICT')
        if row['checkpoint_hash'] != body['parent_hash']:
            reject(409, 'CHECKPOINT_CONFLICT')

    def public_checkpoint(self, row):
        return {'profile_uuid': row['profile_uuid'], 'career_id': row['career_id'],
                'revision': row['revision'], 'checkpoint_hash': row['checkpoint_hash'],
                'snapshot': json.loads(row['snapshot'])}

    def public_delivery(self, row, receipt_state='confirmed'):
        payload = {k: row[k] for k in ('operation_id', 'receipt_id', 'profile_uuid', 'career_id',
                                    'state', 'expected_revision', 'parent_hash', 'candidate_hash')} | {
                'candidate': json.loads(row['candidate'])}
        payload['receipt_state'] = receipt_state
        payload['manual_review'] = receipt_state != 'confirmed'
        if row['state'] != 'prepared' or payload['manual_review']:
            payload.pop('candidate')
        return payload

    def get_checkpoint(self, user, body):
        fields(body, ('profile_uuid', 'career_id'))
        args = self.args(user, body)
        with self.transaction() as conn:
            self.authenticate(conn, user)
            row = self.checkpoint(conn, args)
            if not row:
                reject(404, 'CHECKPOINT_NOT_FOUND')
            return 200, self.public_checkpoint(row)

    def put_checkpoint(self, user, body, key):
        fields(body, ('profile_uuid', 'career_id', 'expected_revision', 'parent_hash', 'snapshot'))
        args = self.args(user, body)
        revision(body['expected_revision']); hash_value(body['parent_hash'], initial=body['expected_revision'] == 0)
        identifier(key)
        request_hash = fingerprint(body)
        with self.transaction() as conn:
            self.authenticate(conn, user)
            row = self.checkpoint(conn, args)
            if row and row['last_key'] == key:
                if row['last_request_hash'] != request_hash:
                    reject(409, 'IDEMPOTENCY_CONFLICT')
                return 200, self.public_checkpoint(row)
            if row:
                self.parent(row, body)
                pending = conn.execute(text('''SELECT 1 FROM steam_wallet_deliveries
                    WHERE user_id=:u AND profile_uuid=:p AND career_id=:c AND state='prepared' LIMIT 1'''), args).first()
                if pending:
                    reject(409, 'DELIVERY_PENDING')
            else:
                if body['expected_revision'] != 0 or body['parent_hash'] is not None:
                    reject(409, 'REVISION_CONFLICT')
                # Existing ownership metadata only; NEVER read the legacy cloud balance.
                owned = conn.execute(text('''SELECT 1 FROM cloud_saves_v2
                    WHERE user_id=:u AND profile_uuid=:p AND career_id=:c LIMIT 1'''), args).first()
                if not owned:
                    reject(409, 'CAREER_NOT_BOUND')
            meta = json.loads(row['snapshot'])['steam_wallet'] if row else None
            blob = self.snapshot(body['snapshot'], args, meta)
            values = args | {'rev': (row['revision'] if row else 0) + 1, 'blob': encoded(blob),
                            'hash': fingerprint(blob), 'key': key, 'req': request_hash}
            if row:
                conn.execute(text('''UPDATE steam_wallet_checkpoints SET revision=:rev,
                    snapshot=:blob,checkpoint_hash=:hash,last_key=:key,last_request_hash=:req
                    WHERE user_id=:u AND profile_uuid=:p AND career_id=:c'''), values)
            else:
                conn.execute(text('''INSERT INTO steam_wallet_checkpoints
                    (user_id,profile_uuid,career_id,revision,snapshot,checkpoint_hash,last_key,last_request_hash)
                    VALUES (:u,:p,:c,:rev,:blob,:hash,:key,:req)'''), values)
            return (200 if row else 201), self.public_checkpoint(self.checkpoint(conn, args))

    def enroll(self, user, body, key):
        fields(body, ('profile_uuid','career_id','snapshot','accept_initial_registration'))
        if self.steam_id is None or body['accept_initial_registration'] is not True:
            reject(403, 'STEAM_REGISTRATION_REQUIRED')
        args = self.args(user, body)
        identifier(key)
        raw = body['snapshot']
        if not isinstance(raw, dict) or raw.get('career_id') != args['c'] or not isinstance(raw.get('profile_id'), str):
            reject(400, 'INVALID_SNAPSHOT')
        if '_steam_local' in raw or 'steam_wallet' in raw or raw.get('l5_match', {}).get('pending') is not None:
            reject(409, 'EXISTING_STEAM_STATE_OR_PENDING_MATCH')
        self.snapshot(raw, args)  # Validate before acquiring registration locks.
        with self.transaction() as conn:
            # Short first-registration lock also serializes ordinary cloud writers.
            # Take it before user locks; never hold it across a Valve/network call.
            conn.execute(text('LOCK TABLE cloud_saves_v2 IN SHARE ROW EXCLUSIVE MODE'))
            self.authenticate(conn, user)
            owners = conn.execute(text('SELECT user_id,profile_uuid FROM cloud_saves_v2 WHERE career_id=:c UNION SELECT user_id,profile_uuid FROM steam_wallet_checkpoints WHERE career_id=:c'), args).all()
            if any(owner[0] != user for owner in owners) or len(owners) > 1:
                reject(409, 'CAREER_BINDING_CONFLICT')
            if owners:
                # Reuse the canonical profile of this user's existing cloud career.
                args['p'] = owners[0][1]
            blob = self.snapshot(raw, args)
            digest = fingerprint(blob)
            row = self.checkpoint(conn, args)
            if row:
                # Lost enrollment response can be recovered, but never reset a wallet.
                has_order = conn.execute(text('SELECT 1 FROM steam_orders WHERE user_id=:u AND profile_uuid=:p AND career_id=:c LIMIT 1'), args).first()
                if has_order or row['revision'] != 1 or row['checkpoint_hash'] != digest:
                    reject(409, 'CHECKPOINT_CONFLICT')
                return 200, self.public_checkpoint(row)
            conn.execute(text('INSERT INTO cloud_saves_v2(save_id,user_id,profile_uuid,career_id) VALUES(:sid,:u,:p,:c) ON CONFLICT DO NOTHING'), args | {'sid':uuid.uuid4().hex})
            request = {'profile_uuid':args['p'],'career_id':args['c'],'expected_revision':0,'parent_hash':None,'snapshot':raw}
            values = args | {'blob':encoded(blob),'hash':digest,'key':key,'req':fingerprint(request)}
            conn.execute(text('INSERT INTO steam_wallet_checkpoints(user_id,profile_uuid,career_id,revision,snapshot,checkpoint_hash,last_key,last_request_hash) VALUES(:u,:p,:c,1,:blob,:hash,:key,:req)'), values)
            return 201, self.public_checkpoint(self.checkpoint(conn,args))

    def receipt(self, conn, rid, args, steam_id):
        value = self.source.lock(conn, rid)
        if value is None:
            reject(404, 'RECEIPT_NOT_FOUND')
        # Source is trusted SERVER code; still fail closed on an incomplete contract.
        required = {'receipt_id', 'user_id', 'profile_uuid', 'career_id', 'steam_id',
                    'app_id', 'environment', 'quantity', 'state'}
        if not isinstance(value, dict) or set(value) != required:
            reject(503, 'STEAM_A4_BAD_RECEIPT')
        if (value['receipt_id'], value['user_id'], value['profile_uuid'], value['career_id'],
            value['steam_id'], value['app_id']) != (rid, args['u'], args['p'], args['c'], steam_id, APP_ID):
            reject(409, 'RECEIPT_DESTINATION_CONFLICT')
        if value['environment'] not in ('sandbox', 'live') or type(value['quantity']) is not int or value['quantity'] not in (25, 75, 150):
            reject(503, 'STEAM_A4_BAD_RECEIPT')
        immutable = {k: v for k, v in value.items() if k != 'state'}
        return value, fingerprint(immutable)

    def create_delivery(self, user, body, key):
        fields(body, ('receipt_id', 'profile_uuid', 'career_id', 'expected_revision', 'parent_hash', 'snapshot'))
        args = self.args(user, body)
        identifier(body['receipt_id']); identifier(key)
        revision(body['expected_revision']); hash_value(body['parent_hash'])
        request_hash = fingerprint(body)
        with self.transaction() as conn:
            steam = self.authenticate(conn, user)
            cp = self.checkpoint(conn, args)
            if not cp:
                reject(409, 'CHECKPOINT_REQUIRED')
            receipt, receipt_hash = self.receipt(conn, body['receipt_id'], args, steam)
            old = conn.execute(text('''SELECT * FROM steam_wallet_deliveries
                WHERE user_id=:u AND idempotency_key=:key'''), args | {'key': key}).mappings().first()
            if old:
                if old['request_hash'] != request_hash or old['receipt_hash'] != receipt_hash:
                    reject(409, 'IDEMPOTENCY_CONFLICT')
                return 200, self.public_delivery(old, receipt['state'])
            exists = conn.execute(text('SELECT 1 FROM steam_wallet_deliveries WHERE receipt_id=:id'), {'id': body['receipt_id']}).first()
            if exists:
                reject(409, 'RECEIPT_ALREADY_RESERVED')
            if receipt['state'] != 'confirmed':
                reject(409, 'RECEIPT_NOT_CONFIRMED')
            self.parent(cp, body)
            pending = conn.execute(text('''SELECT 1 FROM steam_wallet_deliveries WHERE
                user_id=:u AND profile_uuid=:p AND career_id=:c AND state='prepared' LIMIT 1'''), args).first()
            if pending:
                reject(409, 'DELIVERY_PENDING')
            candidate = self.snapshot(body['snapshot'], args, json.loads(cp['snapshot'])['steam_wallet'])
            if body['receipt_id'] in candidate['steam_wallet']['applied_receipts']:
                reject(409, 'RECEIPT_ALREADY_APPLIED')
            operation = str(uuid.uuid4())
            candidate['wallet']['tokens'] += receipt['quantity']
            candidate['steam_wallet']['applied_receipts'][body['receipt_id']] = {
                'receipt_hash': receipt_hash, 'quantity': receipt['quantity'], 'operation_id': operation}
            blob = encoded(candidate)
            if len(blob.encode()) > MAX_BYTES:
                reject(413, 'SNAPSHOT_TOO_LARGE')
            values = args | {'op': operation, 'id': body['receipt_id'], 'key': key, 'req': request_hash,
                'rh': receipt_hash, 'rev': cp['revision'], 'parent': cp['checkpoint_hash'],
                'blob': blob, 'hash': fingerprint(candidate)}
            conn.execute(text('''INSERT INTO steam_wallet_deliveries
                (operation_id,receipt_id,user_id,profile_uuid,career_id,idempotency_key,request_hash,
                 receipt_hash,expected_revision,parent_hash,candidate,candidate_hash,state)
                VALUES (:op,:id,:u,:p,:c,:key,:req,:rh,:rev,:parent,:blob,:hash,'prepared')'''), values)
            return 201, self.public_delivery(conn.execute(text('SELECT * FROM steam_wallet_deliveries WHERE operation_id=:op'), values).mappings().one())

    def locate(self, conn, user, operation):
        identifier(operation)
        row = conn.execute(text('SELECT * FROM steam_wallet_deliveries WHERE operation_id=:op AND user_id=:u'), {'op': operation, 'u': user}).mappings().first()
        if not row:
            reject(404, 'DELIVERY_NOT_FOUND')
        return row

    def get_delivery(self, user, operation):
        with self.transaction() as conn:
            steam = self.authenticate(conn, user)
            row = self.locate(conn, user, operation)
            args = {'u': user, 'p': row['profile_uuid'], 'c': row['career_id']}
            self.checkpoint(conn, args)
            receipt, receipt_hash = self.receipt(conn, row['receipt_id'], args, steam)
            if receipt_hash != row['receipt_hash']:
                reject(409, 'RECEIPT_CONFLICT')
            return 200, self.public_delivery(row, receipt['state'])

    def acknowledge(self, user, operation, body):
        fields(body, ('candidate_hash',)); hash_value(body['candidate_hash'])
        with self.transaction() as conn:
            steam = self.authenticate(conn, user)
            found = self.locate(conn, user, operation)  # immutable routing; no lock yet
            args = {'u': user, 'p': found['profile_uuid'], 'c': found['career_id']}
            cp = self.checkpoint(conn, args)
            receipt, receipt_hash = self.receipt(conn, found['receipt_id'], args, steam)
            row = conn.execute(text('SELECT * FROM steam_wallet_deliveries WHERE operation_id=:op FOR UPDATE'), {'op': operation}).mappings().one()
            if row['candidate_hash'] != body['candidate_hash'] or row['receipt_hash'] != receipt_hash:
                reject(409, 'ACK_CONFLICT')
            old = conn.execute(text('SELECT * FROM steam_wallet_acks WHERE operation_id=:op'), {'op': operation}).mappings().first()
            if old:
                return 200, dict(old)  # historical ACK; never restore its older snapshot
            if receipt['state'] != 'confirmed':
                reject(409, 'RECEIPT_NOT_CONFIRMED')
            if row['state'] != 'prepared':
                reject(409, 'DELIVERY_CONFLICT')
            self.parent(cp, {'expected_revision': row['expected_revision'], 'parent_hash': row['parent_hash']})
            values = args | {'op': operation, 'rev': cp['revision'] + 1,
                            'blob': row['candidate'], 'hash': row['candidate_hash']}
            conn.execute(text('''UPDATE steam_wallet_checkpoints SET revision=:rev,snapshot=:blob,
                checkpoint_hash=:hash,last_key=NULL,last_request_hash=NULL
                WHERE user_id=:u AND profile_uuid=:p AND career_id=:c'''), values)
            conn.execute(text("UPDATE steam_wallet_deliveries SET state='acknowledged' WHERE operation_id=:op"), values)
            conn.execute(text('''INSERT INTO steam_wallet_acks(operation_id,revision,candidate_hash)
                VALUES (:op,:rev,:hash)'''), values)
            return 200, {'operation_id': operation, 'revision': values['rev'], 'candidate_hash': values['hash']}


def create_router(engine_provider, require_bearer, *, receipt_source=None, enabled=None, purchase_sessions=False):
    router = APIRouter()
    active = enabled if enabled is not None else lambda: os.environ.get('STEAM_WALLET_ENABLED', '') == '1'

    async def call(request, action, operation=None):
        try:
            if not active():
                reject(503, 'STEAM_WALLET_DISABLED')
            try:
                claims = require_bearer(request.headers.get('authorization', ''))
            except (TypeError, ValueError, OverflowError):
                reject(401, 'INVALID_SESSION')
            if not isinstance(claims, dict) or not isinstance(claims.get('sub'), str) or not claims['sub']:
                reject(401, 'INVALID_SESSION')
            expiry = claims.get('exp')
            if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= time.time():
                reject(401, 'INVALID_SESSION')
            if request.method == 'GET':
                body = dict(request.query_params)
            else:
                if request.headers.get('content-type', '').split(';')[0].strip() != 'application/json':
                    reject(415, 'JSON_REQUIRED')
                raw = bytearray()
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > MAX_BYTES:
                        reject(413, 'REQUEST_TOO_LARGE')
                def unique_pairs(pairs):
                    out = {}
                    for key, value in pairs:
                        if key in out:
                            reject(400, 'INVALID_JSON')
                        out[key] = value
                    return out
                body = json.loads(raw, object_pairs_hook=unique_pairs)
            user = claims['sub']
            steam_id = claims.get('steam_id') if purchase_sessions and claims.get('typ') == 'steam_purchase' else None
            service = Wallet(engine_provider, receipt_source, steam_id=steam_id)
            if action == 'enroll':
                result = await run_in_threadpool(service.enroll, user, body, request.headers.get('idempotency-key'))
            elif action == 'get':
                result = await run_in_threadpool(service.get_checkpoint, user, body)
            elif action == 'put':
                result = await run_in_threadpool(service.put_checkpoint, user, body, request.headers.get('idempotency-key'))
            elif action == 'create':
                result = await run_in_threadpool(service.create_delivery, user, body, request.headers.get('idempotency-key'))
            elif action == 'status':
                result = await run_in_threadpool(service.get_delivery, user, operation)
            else:
                result = await run_in_threadpool(service.acknowledge, user, operation, body)
            return JSONResponse(result[1], status_code=result[0], headers={'Cache-Control': 'no-store'})
        except Rejected as exc:
            status, code = exc.status, exc.code
        except HTTPException as exc:
            status, code = (401, 'INVALID_SESSION') if exc.status_code == 401 else (503, 'AUTH_UNAVAILABLE')
        except IntegrityError:
            status, code = 409, 'WALLET_CONFLICT'
        except SQLAlchemyError:
            status, code = 503, 'WALLET_STORE_UNAVAILABLE'
        except (ValueError, UnicodeError, RecursionError, OverflowError):
            status, code = 400, 'INVALID_JSON'
        except Exception:
            # Never expose SQL parameters, credentials, adapter payloads or tracebacks.
            status, code = 503, 'WALLET_UNAVAILABLE'
        return JSONResponse({'detail': code}, status_code=status, headers={'Cache-Control': 'no-store'})

    if purchase_sessions:
        @router.post('/v1/steam/wallet/enroll')
        async def enroll(request: Request):
            return await call(request, 'enroll')

    @router.get('/v1/steam/wallet/checkpoint')
    async def get(request: Request):
        return await call(request, 'get')

    @router.put('/v1/steam/wallet/checkpoint')
    async def put(request: Request):
        return await call(request, 'put')

    @router.post('/v1/steam/wallet/deliveries')
    async def create(request: Request):
        return await call(request, 'create')

    @router.get('/v1/steam/wallet/deliveries/{operation_id}')
    async def status(request: Request, operation_id: str):
        return await call(request, 'status', operation_id)

    @router.post('/v1/steam/wallet/deliveries/{operation_id}/ack')
    async def ack(request: Request, operation_id: str):
        return await call(request, 'ack', operation_id)

    return router
