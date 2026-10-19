"""Isolated Steam identity linking. Schema provisioned manually; no startup I/O."""
import json
import re
import os
import uuid
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from dataclasses import dataclass
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

APP_ID = 5069970
IDENTITY = 'basketcorp-identity-v1'
VALVE_URL = 'https://partner.steam-api.com/ISteamUserAuth/AuthenticateUserTicket/v1/'

class Rejected(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code

@dataclass(frozen=True, repr=False)
class ValveRequest:
    url: str
    appid: int
    identity: str
    ticket: str
    timeout_seconds: int = 5

class Store:
    def __init__(self, engine_provider):
        self.engine_provider = engine_provider

    def engine(self):
        engine = self.engine_provider()
        if engine is None:
            raise Rejected(503, 'IDENTITY_STORE_UNAVAILABLE')
        return engine

    def eligible(self, conn, user, lock=False):
        query = 'SELECT status,email_verified FROM users WHERE user_id=:user'
        if lock and conn.dialect.name == 'postgresql':
            query += ' FOR UPDATE'
        row = conn.execute(text(query), {'user': user}).first()
        if row is None:
            raise Rejected(401, 'SESSION_ACCOUNT_MISSING')
        if row[0] != 'active' or not row[1]:
            raise Rejected(403, 'VERIFIED_ACTIVE_ACCOUNT_REQUIRED')

    def check(self, user):
        with self.engine().connect() as conn:
            self.eligible(conn, user)

    def link(self, user, steamid):
        with self.engine().begin() as conn:
            self.eligible(conn, user, lock=True)
            args = {'user': user, 'steam': steamid, 'appid': APP_ID, 'identity': IDENTITY}
            inserted = conn.execute(text("""
                INSERT INTO steam_identity_links(user_id,steam_id,app_id,service_identity)
                VALUES(:user,:steam,:appid,:identity)
                ON CONFLICT DO NOTHING RETURNING user_id
            """), args).first()
            if inserted:
                return True
            rows = conn.execute(text("""
                SELECT user_id,steam_id FROM steam_identity_links
                WHERE user_id=:user OR steam_id=:steam
            """), args).all()
            if len(rows) == 1 and tuple(rows[0]) == (user,steamid):
                return False
            raise Rejected(409, 'STEAM_LINK_CONFLICT')

    def eligible_steam(self, conn, user, steamid, lock=False):
        query = 'SELECT status FROM users WHERE user_id=:u'
        if lock and conn.dialect.name == 'postgresql':
            query += ' FOR UPDATE'
        row = conn.execute(text(query), {'u':user}).first()
        if not row or row[0] != 'active':
            raise Rejected(403, 'ACTIVE_ACCOUNT_REQUIRED')
        link = conn.execute(text('SELECT steam_id,app_id,service_identity FROM steam_identity_links WHERE user_id=:u'), {'u':user}).first()
        if not link or tuple(link) != (steamid, APP_ID, IDENTITY):
            raise Rejected(409, 'STEAM_LINK_CONFLICT')

    def purchase_user(self, steamid):
        # steamid MUST be obtained from Service.verify_ticket, never request input.
        with self.engine().begin() as conn:
            if conn.dialect.name != 'postgresql':
                raise Rejected(503, 'STEAM_WALLET_POSTGRES_REQUIRED')
            conn.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:sid,0))'), {'sid':'steam_identity:'+steamid})
            row = conn.execute(text('SELECT user_id FROM steam_identity_links WHERE steam_id=:sid'), {'sid':steamid}).first()
            if row:
                self.eligible_steam(conn, row[0], steamid, lock=True)
                return row[0]
            user = uuid.uuid4().hex
            conn.execute(text("INSERT INTO users(user_id,status,email_verified,email) VALUES(:u,'active',FALSE,:email)"),
                         {'u':user,'email':'steam_'+user+'@identity.invalid'})
            conn.execute(text('INSERT INTO steam_identity_links(user_id,steam_id,app_id,service_identity) VALUES(:u,:sid,:app,:identity)'),
                         {'u':user,'sid':steamid,'app':APP_ID,'identity':IDENTITY})
            return user

class Service:
    def __init__(self, store, valve_transport):
        self.store, self.valve_transport = store, valve_transport

    def verify_ticket(self, body):
        if not isinstance(body, dict) or set(body) != {'ticket'}:
            raise Rejected(400, 'INVALID_REQUEST')
        ticket = body['ticket']
        if not isinstance(ticket, str) or not 2 <= len(ticket) <= 8192 or len(ticket) % 2 or not re.fullmatch('[0-9a-fA-F]+',ticket):
            raise Rejected(400, 'INVALID_TICKET_FORMAT')
        try:
            status, data = self.valve_transport(ValveRequest(VALVE_URL, APP_ID, IDENTITY, ticket))
        except Exception:
            raise Rejected(503, 'STEAM_UNAVAILABLE') from None
        if status != 200:
            raise Rejected(503, 'STEAM_UNAVAILABLE')
        if not isinstance(data,dict) or not isinstance(data.get('response'),dict):
            raise Rejected(502, 'STEAM_BAD_RESPONSE')
        response = data['response']
        if 'error' in response:
            error = response['error']
            if isinstance(error,dict) and error.get('errorcode') == 101:
                raise Rejected(401, 'STEAM_TICKET_REJECTED')
            raise Rejected(503, 'STEAM_UNAVAILABLE')
        params = response.get('params')
        if not isinstance(params,dict) or params.get('result') != 'OK':
            raise Rejected(502, 'STEAM_BAD_RESPONSE')
        # Valve validates the configured appid/identity; they are NOT documented echoes.
        # If an echo is present, fail closed on mismatch.
        if ('appid' in params and params['appid'] != APP_ID) or ('identity' in params and params['identity'] != IDENTITY):
            raise Rejected(502, 'STEAM_CONTEXT_MISMATCH')
        steamid = params.get('steamid')
        if not isinstance(steamid,str) or not re.fullmatch('[1-9][0-9]{0,19}',steamid) or int(steamid) >= 2**64:
            raise Rejected(502, 'STEAM_BAD_RESPONSE')
        return steamid

    def link(self, user, body):
        self.store.check(user)
        steamid = self.verify_ticket(body)
        created = self.store.link(user, steamid)
        return (201 if created else 200), {'linked':True,'app_id':APP_ID,'status':'linked' if created else 'already_linked'}

def create_router(engine_provider, require_bearer, valve_transport=None, enabled=None):
    app = APIRouter()
    store = Store(engine_provider)
    if valve_transport is None:
        valve_transport = ValveAdapter(lambda: os.environ.get('STEAM_PUBLISHER_KEY', ''))
    if enabled is None:
        enabled = lambda: os.environ.get('STEAM_IDENTITY_ENABLED', '') == '1'
    service = Service(store,valve_transport)

    @app.post('/v1/steam/identity/link')
    async def link(request: Request):
        try:
            if not enabled():
                raise Rejected(503, 'STEAM_IDENTITY_DISABLED')
            # Production adapter reuses _require_bearer_claims without altering it.
            claims = require_bearer(request.headers.get('authorization',''))
            user = claims.get('sub')
            if not isinstance(user,str) or not user:
                raise Rejected(401,'INVALID_SESSION')
            # Existing helper checks expiry when present; enforce it for this new route.
            if not isinstance(claims.get('exp'), (int,float)):
                raise Rejected(401,'INVALID_SESSION')
            if request.headers.get('content-type','').split(';')[0].strip() != 'application/json':
                raise Rejected(415,'JSON_REQUIRED')
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw)>10000:
                    raise Rejected(413,'REQUEST_TOO_LARGE')
            try:
                body = json.loads(raw)
            except (ValueError,UnicodeError):
                raise Rejected(400,'INVALID_JSON') from None
            status, payload = await run_in_threadpool(service.link, user, body)
            return JSONResponse(payload,status_code=status,headers={'Cache-Control':'no-store'})
        except Rejected as err:
            return JSONResponse({'detail':err.code},status_code=err.status,headers={'Cache-Control':'no-store'})
        except HTTPException as err:
            # Do not echo exception details or request input from authentication.
            code = 'INVALID_SESSION' if err.status_code == 401 else 'AUTH_UNAVAILABLE'
            return JSONResponse({'detail':code},status_code=401 if err.status_code == 401 else 503,headers={'Cache-Control':'no-store'})
        except SQLAlchemyError:
            return JSONResponse({'detail':'IDENTITY_STORE_UNAVAILABLE'},status_code=503,headers={'Cache-Control':'no-store'})
    return app


import json
import ssl
import certifi
import urllib.request
import urllib.parse

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class ValveAdapter:
    def __init__(self, key_provider, opener=None):
        self.key_provider = key_provider
        # Verify Valve certificates and hostname using the installed CA bundle.
        # No redirects carrying credentials.
        self.opener = opener or urllib.request.build_opener(
            NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())),
        )

    def __call__(self, request):
        if (request.url,request.appid,request.identity) != (VALVE_URL, APP_ID, IDENTITY):
            raise RuntimeError('STEAM_CONTEXT_MISMATCH')
        try:
            key = self.key_provider()
            if not isinstance(key,str) or not key:
                raise ValueError()
            query = urllib.parse.urlencode({'key':key,'appid':APP_ID,'identity':IDENTITY,'ticket':request.ticket})
            req = urllib.request.Request(VALVE_URL+'?'+query,method='GET',headers={'Accept':'application/json'})
            with self.opener.open(req,timeout=request.timeout_seconds) as response:
                if response.status != 200:
                    return response.status, {}
                # Bounded read. HTTP/socket timeout, not a strict total wall-clock deadline.
                body=response.read(65537)
                if len(body)>65536:
                    raise ValueError()
                return 200,json.loads(body)
        except Exception:
            # Never propagate URL, ticket, key or upstream body in exceptions.
            raise RuntimeError('STEAM_UNAVAILABLE') from None
