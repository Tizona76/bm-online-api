"""Synthetic tests only. No production main import, network call or migration.
PG tests require A5_TEST_CLUSTER pointing at a private /private/tmp cluster made
for this task, with marker A5_TEST_ONLY, socket-only PostgreSQL and user a5_test.
The preparatory SQL file is NEVER executed; fixture DDL is isolated per test.
"""
import ast
import asyncio
from urllib.parse import urlsplit
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import secrets
import sys
import time
import types
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from typing import Dict, Any

from fastapi import FastAPI, Header, HTTPException
from jose import jwt, JWTError
from sqlalchemy import create_engine, event, text

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))
import steam_wallet as sw

# Extract unchanged JWT functions only. Never run main.py startup or env loading.
main_source = (API / 'main.py').read_text()
nodes = [n for n in ast.parse(main_source).body if isinstance(n, ast.FunctionDef)
         and n.name in ('_jwt_verify_access', '_require_bearer_claims')]
auth = types.ModuleType('wallet_test_auth')
auth.__dict__.update(Dict=Dict, Any=Any, Header=Header, HTTPException=HTTPException,
                     jwt=jwt, JWTError=JWTError, JWT_SECRET=secrets.token_hex(32),
                     JWT_ISSUER='a5-test', JWT_AUDIENCE='a5-test')
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<test JWT helpers>', 'exec'), auth.__dict__)


def token(user='u1', **extra):
    claims = {'sub': user, 'typ': 'access', 'iss': 'a5-test', 'aud': 'a5-test', 'exp': time.time()+60}
    claims.update(extra)
    return jwt.encode(claims, auth.JWT_SECRET, algorithm='HS256')


class ASGIClient:
    """Exercise the real ASGI routes without installing an HTTP client library."""
    def __init__(self, app): self.app = app
    def close(self): pass
    def get(self, path, **kw): return self.request('GET', path, **kw)
    def put(self, path, **kw): return self.request('PUT', path, **kw)
    def request(self, method, path, json=None, content=None, headers=None):
        url = urlsplit(path)
        hdr = dict(headers or {})
        if content is None:
            content = __import__('json').dumps(json)
            hdr.setdefault('Content-Type', 'application/json')
        data = content.encode() if isinstance(content, str) else content
        messages = []
        async def invoke():
            async def receive(): return {'type':'http.request','body':data,'more_body':False}
            async def send(message): messages.append(message)
            await self.app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
                'method':method,'scheme':'https','path':url.path,'raw_path':url.path.encode(),
                'query_string':url.query.encode(),'root_path':'',
                'headers':[(k.lower().encode(),v.encode()) for k,v in hdr.items()],
                'server':('test',443),'client':('test',1)},receive,send)
        asyncio.run(invoke())
        body=b''.join(m.get('body',b'') for m in messages).decode()
        return types.SimpleNamespace(status_code=messages[0]['status'],text=body,
                                     json=lambda:__import__('json').loads(body))


def client(engine=lambda: None, source=None, enabled=None):
    app = FastAPI()
    @app.get('/existing')
    def existing(): return {'untouched': True}
    app.include_router(sw.create_router(engine, auth._require_bearer_claims,
                                      receipt_source=source, enabled=enabled))
    return ASGIClient(app)


class RouteTests(unittest.TestCase):
    def test_disabled_all_routes_without_io(self):
        with patch.dict(os.environ, {}, clear=True):
            c = client(lambda: self.fail('database touched'))
            for method, url in [('GET','checkpoint'),('PUT','checkpoint'),('POST','deliveries'),
                                ('GET','deliveries/unknown'),('POST','deliveries/unknown/ack')]:
                r = c.request(method,'/v1/steam/wallet/'+url)
                self.assertEqual(r.status_code,503)
                self.assertEqual(r.json()['detail'],'STEAM_WALLET_DISABLED')
    def test_absent_jwt(self):
        self.assertEqual(client(enabled=lambda:True).get('/v1/steam/wallet/checkpoint').status_code,401)
    def test_invalid_jwt(self):
        self.assertEqual(client(enabled=lambda:True).get('/v1/steam/wallet/checkpoint',headers={'Authorization':'Bearer invalid'}).status_code,401)
    def test_expired_or_missing_expiry(self):
        for exp in (None,False,time.time()-1):
            r=client(enabled=lambda:True).get('/v1/steam/wallet/checkpoint',headers={'Authorization':'Bearer '+token(exp=exp)})
            self.assertEqual(r.status_code,401)
    def test_import_has_no_io(self):
        with patch('sqlalchemy.create_engine',side_effect=AssertionError('DB')), patch('socket.socket.connect',side_effect=AssertionError('NETWORK')):
            importlib.reload(sw)
            sw.create_router(lambda:self.fail('provider'),auth._require_bearer_claims)
    def test_existing_route_unchanged(self):
        self.assertEqual(client().get('/existing').json(),{'untouched':True})
        tree=ast.parse(main_source)
        self.assertEqual(len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_steam_wallet_router']),1)
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_steam_wallet_router')
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}, {'purchase_sessions': True}) # no test receipt source wired in production
    def test_invalid_request_redacted(self):
        c=client(enabled=lambda:True)
        r=c.put('/v1/steam/wallet/checkpoint',content='{"secret":"synthetic-needle","secret":2}',headers={'Authorization':'Bearer '+token(),'Content-Type':'application/json'})
        self.assertEqual(r.status_code,400);self.assertNotIn('synthetic-needle',r.text)


class ReceiptFixture:
    """Only in the test file; never importable via any API request."""
    def lock(self, conn, rid):
        row=conn.execute(text('SELECT payload FROM receipt_fixture WHERE receipt_id=:r FOR UPDATE'),{'r':rid}).first()
        return json.loads(row[0]) if row else None


FIXTURE_SQL = '''
CREATE TABLE users(user_id TEXT PRIMARY KEY,status TEXT,email_verified BOOLEAN);
CREATE TABLE steam_identity_links(user_id TEXT UNIQUE REFERENCES users,steam_id TEXT UNIQUE,app_id INT,service_identity TEXT);
CREATE TABLE cloud_saves_v2(user_id TEXT,profile_uuid TEXT,career_id TEXT);
CREATE TABLE receipt_fixture(receipt_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE steam_wallet_checkpoints(user_id TEXT REFERENCES steam_identity_links(user_id),profile_uuid TEXT,career_id TEXT,
 revision BIGINT CHECK(revision>0),snapshot TEXT,checkpoint_hash TEXT,last_key TEXT,last_request_hash TEXT,
 PRIMARY KEY(user_id,profile_uuid,career_id));
CREATE TABLE steam_wallet_deliveries(operation_id TEXT PRIMARY KEY,receipt_id TEXT UNIQUE REFERENCES receipt_fixture,
 user_id TEXT,profile_uuid TEXT,career_id TEXT,idempotency_key TEXT,request_hash TEXT,receipt_hash TEXT,
 expected_revision BIGINT,parent_hash TEXT,candidate TEXT,candidate_hash TEXT,state TEXT CHECK(state IN ('prepared','acknowledged')),
 UNIQUE(user_id,idempotency_key),UNIQUE(operation_id,candidate_hash),
 FOREIGN KEY(user_id,profile_uuid,career_id) REFERENCES steam_wallet_checkpoints);
CREATE UNIQUE INDEX one_pending ON steam_wallet_deliveries(user_id,profile_uuid,career_id) WHERE state='prepared';
CREATE TABLE steam_wallet_acks(operation_id TEXT PRIMARY KEY,revision BIGINT,candidate_hash TEXT,
 FOREIGN KEY(operation_id,candidate_hash) REFERENCES steam_wallet_deliveries(operation_id,candidate_hash));
'''


@unittest.skipUnless(os.environ.get('A5_TEST_CLUSTER'), 'private PostgreSQL cluster not supplied')
class PostgreSQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        cls.root=Path(os.environ['A5_TEST_CLUSTER']).resolve()
        if cls.root.parent != Path('/private/tmp') or not (cls.root/'A5_TEST_ONLY').is_file() or cls.root.stat().st_mode&0o777 != 0o700:
            raise RuntimeError('unverified isolated cluster')
        # Discard inherited PG configuration WITHOUT reading any credentials.
        for key in list(os.environ):
            if key.startswith('PG'):os.environ.pop(key,None)
        cls.kw={'host':str(cls.root/'socket'),'port':55486,'user':'a5_test','dbname':'postgres',
                'passfile':str(cls.root/'absent'),'connect_timeout':3}
        with psycopg.connect(**cls.kw) as c:
            row=c.execute("SELECT current_setting('data_directory'),current_setting('listen_addresses'),inet_server_addr(),version()").fetchone()
            if row[0]!=str(cls.root/'data') or row[1]!='' or row[2] is not None or not row[3].startswith('PostgreSQL 18.6 '):
                raise RuntimeError('isolation not demonstrated')
    def setUp(self):
        import psycopg
        self.schema='a5_'+uuid.uuid4().hex
        with psycopg.connect(**self.kw) as c:c.execute('CREATE SCHEMA '+self.schema)
        self.engine=create_engine('postgresql+psycopg://',creator=lambda:psycopg.connect(**self.kw,options='-c search_path='+self.schema+' -c lock_timeout=5000 -c statement_timeout=10000'),hide_parameters=True)
        with self.engine.begin() as c:
            for statement in FIXTURE_SQL.split(';'):
                if statement.strip():c.execute(text(statement))
            c.execute(text("INSERT INTO users VALUES ('u1','active',TRUE),('u2','active',TRUE),('guest','active',FALSE)"))
            c.execute(text("INSERT INTO steam_identity_links VALUES ('u1','76561198000000001',5069970,'basketcorp-identity-v1'),('guest','76561198000000002',5069970,'basketcorp-identity-v1')"))
            c.execute(text("INSERT INTO cloud_saves_v2 VALUES ('u1','profile1','career1'),('u1','profile1','career2')"))
        self.c=client(lambda:self.engine,ReceiptFixture(),lambda:True)
        self.base={'profile_uuid':'profile1','career_id':'career1','expected_revision':0,'parent_hash':None,
                   'snapshot':{'career_id':'career1','wallet':{'tokens':10},'season_tokens_earned':7}}
        response=self.req('PUT','checkpoint',self.base,'initial');self.assertEqual(response.status_code,201,response.text)
        self.cp=response.json();self.issue()
    def tearDown(self):
        import psycopg
        self.c.close();self.engine.dispose()
        with psycopg.connect(**self.kw) as c:c.execute('DROP SCHEMA '+self.schema+' CASCADE')
    def req(self,method,path,body=None,key=None,user='u1',c=None):
        headers={'Authorization':'Bearer '+token(user)}
        if key:headers['Idempotency-Key']=key
        return (c or self.c).request(method,'/v1/steam/wallet/'+path,json=body,headers=headers)
    def issue(self,rid='receipt1',**changes):
        r={'receipt_id':rid,'user_id':'u1','profile_uuid':'profile1','career_id':'career1',
           'steam_id':'76561198000000001','app_id':5069970,'environment':'sandbox','quantity':25,'state':'confirmed'}
        r.update(changes)
        with self.engine.begin() as c:c.execute(text('INSERT INTO receipt_fixture VALUES (:id,:p) ON CONFLICT(receipt_id) DO UPDATE SET payload=:p'),{'id':rid,'p':json.dumps(r)})
    def payload(self,**kw):
        d={'receipt_id':'receipt1','profile_uuid':'profile1','career_id':'career1',
           'expected_revision':self.cp['revision'],'parent_hash':self.cp['checkpoint_hash'],'snapshot':copy.deepcopy(self.cp['snapshot'])}
        d.update(kw);return d
    def prepare(self,**kw):
        r=self.req('POST','deliveries',self.payload(**kw),'delivery1');self.assertEqual(r.status_code,201,r.text);return r.json()
    def ack(self,d):return self.req('POST','deliveries/'+d['operation_id']+'/ack',{'candidate_hash':d['candidate_hash']})
    def current(self):
        r=self.req('GET','checkpoint?profile_uuid=profile1&career_id=career1');self.assertEqual(r.status_code,200);return r.json()
    def count(self,table):
        with self.engine.connect() as c:return c.execute(text('SELECT count(*) FROM '+table)).scalar()
    def test_a3_missing(self):
        self.assertEqual(self.req('GET','checkpoint?profile_uuid=profile1&career_id=career1',user='u2').status_code,409)
    def test_guest_refused(self):
        self.assertEqual(self.req('GET','checkpoint?profile_uuid=profile1&career_id=career1',user='guest').status_code,403)
    def test_unbound_career(self):
        b=copy.deepcopy(self.base);b['career_id']='not-bound';b['snapshot']['career_id']='not-bound'
        self.assertEqual(self.req('PUT','checkpoint',b,'new').status_code,409)
    def test_wrong_destination_receipt(self):
        self.issue(user_id='u2');self.assertEqual(self.req('POST','deliveries',self.payload(),'x').status_code,409)
    def test_nonexistent_receipt(self):
        self.assertEqual(self.req('POST','deliveries',self.payload(receipt_id='absent'),'x').status_code,404)
    def test_unconfirmed_receipts(self):
        for state in ('pending','approved','revoked','partial_refund','manual'):
            self.issue(state=state);self.assertEqual(self.req('POST','deliveries',self.payload(),'x').status_code,409)
        self.assertEqual(self.count('steam_wallet_deliveries'),0)
    def test_unique_credit(self):
        d=self.prepare();self.assertEqual(d['candidate']['wallet']['tokens'],35)
        self.assertEqual(self.current()['snapshot']['wallet']['tokens'],10) # prepare is not ACK
        self.assertEqual(self.ack(d).status_code,200)
        cp=self.current();self.assertEqual(cp['snapshot']['wallet']['tokens'],35)
        self.assertEqual(cp['snapshot']['season_tokens_earned'],7)
        self.assertEqual(len(cp['snapshot']['steam_wallet']['applied_receipts']),1)
    def test_repeated_receipt_and_key(self):
        d=self.prepare();self.assertEqual(self.req('POST','deliveries',self.payload(),'delivery1').json()['operation_id'],d['operation_id'])
        self.assertEqual(self.req('POST','deliveries',self.payload(),'other-key').status_code,409)
        self.assertEqual(self.count('steam_wallet_deliveries'),1)
    def test_modified_receipt(self):
        d=self.prepare();self.issue(quantity=75);self.assertEqual(self.ack(d).status_code,409)
        self.assertEqual(self.current()['snapshot']['wallet']['tokens'],10)
    def test_concurrent_prepares_and_acks(self):
        def create(_):return self.req('POST','deliveries',self.payload(),'same')
        with ThreadPoolExecutor(2) as ex:responses=list(ex.map(create,range(2)))
        self.assertEqual(sorted(r.status_code for r in responses),[200,201]);d=responses[0].json()
        with ThreadPoolExecutor(2) as ex:responses=list(ex.map(lambda _:self.ack(d),range(2)))
        self.assertTrue(all(r.status_code==200 for r in responses));self.assertEqual(self.count('steam_wallet_acks'),1)
        self.assertEqual(self.current()['snapshot']['wallet']['tokens'],35)
    def test_obsolete_revision(self):
        self.assertEqual(self.req('POST','deliveries',self.payload(expected_revision=0),'x').status_code,409)
    def test_divergent_hash(self):
        self.assertEqual(self.req('POST','deliveries',self.payload(parent_hash='0'*64),'x').status_code,409)
    def test_repeated_ack(self):
        d=self.prepare();a=self.ack(d);b=self.ack(d);self.assertEqual(a.json(),b.json());self.assertEqual(self.count('steam_wallet_acks'),1)
    def test_rollback_before_commit(self):
        d=self.prepare()
        def abort(conn):raise RuntimeError('synthetic-before-commit')
        event.listen(self.engine,'commit',abort)
        try:self.assertEqual(self.ack(d).status_code,503)
        finally:event.remove(self.engine,'commit',abort)
        self.assertEqual(self.count('steam_wallet_acks'),0);self.assertEqual(self.current()['snapshot']['wallet']['tokens'],10)
        self.assertEqual(self.ack(d).status_code,200)
    def test_after_commit_response_lost(self):
        d=self.prepare()
        # Actual transaction commits, then simulate the response serialization failure.
        with patch.object(sw,'JSONResponse',side_effect=[RuntimeError('lost response'),sw.JSONResponse({'detail':'WALLET_UNAVAILABLE'},status_code=503)]):
            self.assertEqual(self.ack(d).status_code,503)
        self.assertEqual(self.ack(d).status_code,200);self.assertEqual(self.current()['snapshot']['wallet']['tokens'],35)
    def test_late_ack_never_overwrites_new_checkpoint(self):
        d=self.prepare();self.ack(d);cp=self.current();snapshot=cp['snapshot'];snapshot['wallet']['tokens']-=6
        body={'profile_uuid':'profile1','career_id':'career1','expected_revision':cp['revision'],'parent_hash':cp['checkpoint_hash'],'snapshot':snapshot}
        self.assertEqual(self.req('PUT','checkpoint',body,'spend').status_code,200)
        self.assertEqual(self.ack(d).status_code,200);self.assertEqual(self.current()['snapshot']['wallet']['tokens'],29)
    def test_refund_before_ack(self):
        d=self.prepare();self.issue(state='revoked');self.assertEqual(self.ack(d).status_code,409)
        r=self.req('GET','deliveries/'+d['operation_id']).json();self.assertTrue(r['manual_review']);self.assertNotIn('candidate',r)
        self.assertEqual(self.current()['snapshot']['wallet']['tokens'],10)
    def test_refund_after_ack_no_debit(self):
        d=self.prepare();self.ack(d);self.issue(state='partial_refund');self.assertEqual(self.ack(d).status_code,200)
        self.assertEqual(self.current()['snapshot']['wallet']['tokens'],35)
    def test_post_retry_after_refund_never_returns_candidate(self):
        d=self.prepare();self.ack(d);self.issue(state='partial_refund')
        r=self.req('POST','deliveries',self.payload(),'delivery1')
        self.assertEqual(r.status_code,200);self.assertTrue(r.json()['manual_review'])
        self.assertNotIn('candidate',r.json());self.assertEqual(self.current()['snapshot']['wallet']['tokens'],35)
    def test_acknowledged_get_never_returns_candidate(self):
        d=self.prepare();self.ack(d)
        r=self.req('GET','deliveries/'+d['operation_id'])
        self.assertEqual(r.status_code,200);self.assertNotIn('candidate',r.json())
    def test_unknown_operation(self):self.assertEqual(self.req('GET','deliveries/unknown').status_code,404)
    def test_free_gain_and_spend_preserved(self):
        blob=self.payload()['snapshot'];blob['wallet']['tokens']+=9-6;blob['season_tokens_earned']+=9
        d=self.prepare(snapshot=blob);self.ack(d);cp=self.current()['snapshot']
        self.assertEqual(cp['wallet']['tokens'],38);self.assertEqual(cp['season_tokens_earned'],16)
    def test_markers_cannot_be_removed(self):
        d=self.prepare();self.ack(d);cp=self.current();blob=cp['snapshot'];del blob['steam_wallet']
        b={'profile_uuid':'profile1','career_id':'career1','expected_revision':cp['revision'],'parent_hash':cp['checkpoint_hash'],'snapshot':blob}
        self.assertEqual(self.req('PUT','checkpoint',b,'remove').status_code,409)
    def test_put_blocked_while_prepared(self):
        self.prepare();b=self.base|{'expected_revision':self.cp['revision'],'parent_hash':self.cp['checkpoint_hash'],'snapshot':self.cp['snapshot']}
        self.assertEqual(self.req('PUT','checkpoint',b,'new').status_code,409)
    def test_no_production_receipt_source(self):
        c=client(lambda:self.engine,enabled=lambda:True)
        self.assertEqual(self.req('POST','deliveries',self.payload(),'x',c=c).status_code,503)
        self.assertEqual(self.count('steam_wallet_deliveries'),0)
    def test_external_fake_injection_rejected(self):
        for k in ('quantity','price','steam_id','receipt_source','confirmed'):
            self.assertEqual(self.req('POST','deliveries',self.payload(**{k:25}),'x').status_code,400)
    def test_error_redaction(self):
        out=io.StringIO()
        with redirect_stdout(out),redirect_stderr(out),patch.object(ReceiptFixture,'lock',side_effect=RuntimeError('secret-synthetic-needle')):
            r=self.req('POST','deliveries',self.payload(),'x')
        self.assertEqual(r.status_code,503);self.assertNotIn('secret-synthetic-needle',r.text+out.getvalue())
    def test_checkpoint_idempotence(self):
        self.assertEqual(self.req('PUT','checkpoint',self.base,'initial').status_code,200)
        b=copy.deepcopy(self.base);b['snapshot']['wallet']['tokens']=99
        self.assertEqual(self.req('PUT','checkpoint',b,'initial').status_code,409)
    def test_no_sql_migration_executed(self):
        sql=(API/'sql'/'steam_wallet_a5.sql').read_text()
        self.assertIn("RAISE EXCEPTION 'A5_MIGRATION_BLOCKED_PENDING_A4_RECEIPT_FOREIGN_KEY'",sql)
        self.assertNotIn('CREATE TABLE', (API/'steam_wallet.py').read_text())

if __name__=='__main__':unittest.main()
