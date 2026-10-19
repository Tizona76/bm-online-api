import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import secrets
import sqlite3
import tempfile
import time
from threading import Barrier
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import ast
import types
import sys
from typing import Dict, Any
from fastapi import FastAPI, Header, HTTPException
from jose import jwt, JWTError
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import steam_identity
# Extract only the existing pure JWT helpers; never import production main.py.
main_source = (Path(__file__).resolve().parents[1] / 'main.py').read_text()
functions = [n for n in ast.parse(main_source).body if isinstance(n,ast.FunctionDef)
             and n.name in {'_jwt_verify_access','_require_bearer_claims'}]
auth = types.ModuleType('auth_fixture')
auth.__dict__.update(Dict=Dict, Any=Any, Header=Header, HTTPException=HTTPException,
                     jwt=jwt, JWTError=JWTError)
exec(compile(ast.Module(body=functions,type_ignores=[]),'<jwt helpers>','exec'),auth.__dict__)
from steam_identity import APP_ID, IDENTITY, VALVE_URL, Store, Service, Rejected, create_router

ROOT=Path(__file__).parent
STEAM='76561198000000001'  # Synthetic fixture, never sent to Valve.

def ok(steam=STEAM, **extra):
    return 200, {'response':{'params':{'result':'OK','steamid':steam,**extra}}}

async def invoke(app, token, body, raw=None):
    data=json.dumps(body).encode() if raw is None else raw
    messages=[]
    async def receive():
        return {'type':'http.request','body':data,'more_body':False}
    async def send(message): messages.append(message)
    headers=[(b'content-type',b'application/json')]
    if token: headers.append((b'authorization',('Bearer '+token).encode()))
    await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST','scheme':'https','path':'/v1/steam/identity/link','raw_path':b'/v1/steam/identity/link','query_string':b'','root_path':'','headers':headers,'server':('test',443),'client':('test',1)},receive,send)
    return messages[0]['status'],json.loads(b''.join(m.get('body',b'') for m in messages))

class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='test-db-')
        self.path=Path(self.tmp.name)/'test.sqlite'
        self.engine=create_engine("sqlite:///"+str(self.path),connect_args={"timeout":5})
        self.store=Store(lambda:self.engine)
        with sqlite3.connect(self.path, factory=ClosingConnection) as c:
            c.executescript('CREATE TABLE users(user_id TEXT PRIMARY KEY,status TEXT,email_verified INTEGER); CREATE TABLE steam_identity_links(steam_id TEXT PRIMARY KEY,user_id TEXT UNIQUE NOT NULL REFERENCES users(user_id),app_id INTEGER NOT NULL CHECK(app_id=5069970),service_identity TEXT NOT NULL,linked_at TEXT DEFAULT CURRENT_TIMESTAMP);')
            c.executemany('INSERT INTO users VALUES(?,?,?)',[('u1','active',1),('u2','active',1),('guest','active',0),('disabled','disabled',1)])
        auth.JWT_SECRET=secrets.token_hex(32) # ephemeral JWT test signing material, not Steam
        auth.JWT_ISSUER='a3-test'; auth.JWT_AUDIENCE='a3-test'
        self.calls=[]; self.reply=ok()
        self.app=FastAPI()
        self.app.include_router(create_router(lambda:self.engine,auth._require_bearer_claims,self.valve,lambda:True))
        # The prototype and tests cannot access network, even accidentally.
        self.network=patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN'))
        self.network.start()
    def tearDown(self):
        self.network.stop();self.engine.dispose();self.tmp.cleanup()
    def token(self,user='u1',**kw):
        claims={'sub':user,'iss':'a3-test','aud':'a3-test','typ':'access','exp':int(time.time())+60}
        claims.update(kw)
        return auth.jwt.encode(claims,auth.JWT_SECRET,algorithm='HS256')
    def valve(self,req):
        self.calls.append(req)
        if isinstance(self.reply,Exception): raise self.reply
        return self.reply
    def request(self,user='u1',body=None,token=None,raw=None):
        return asyncio.run(invoke(self.app,self.token(user) if token is None else token,{'ticket':'ab12'} if body is None else body,raw))
    def count(self):
        with sqlite3.connect(self.path, factory=ClosingConnection) as c:return c.execute('SELECT count(*) FROM steam_identity_links').fetchone()[0]
    def test_link_idempotent_and_persistent(self):
        self.assertEqual(self.request()[0],201)
        self.assertEqual(self.request()[0],200)
        self.assertEqual(self.count(),1)
        with sqlite3.connect(self.path, factory=ClosingConnection) as c:self.assertEqual(c.execute('SELECT user_id FROM steam_identity_links').fetchone()[0],'u1')
    def test_session_required(self):
        for token in ['',self.token(exp=1),self.token(aud='wrong'),self.token(typ='refresh'),'broken']:
            with self.subTest(token_type='invalid'):
                self.assertEqual(self.request(token=token)[0],401)
        self.assertFalse(self.calls)
    def test_guest_disabled_missing(self):
        for user,status in [('guest',403),('disabled',403),('missing',401)]:
            self.assertEqual(self.request(user)[0],status)
        self.assertFalse(self.calls)
    def test_no_expiry(self):
        claims={'sub':'u1','typ':'access','iss':'a3-test','aud':'a3-test'}
        token=auth.jwt.encode(claims,auth.JWT_SECRET,algorithm='HS256')
        self.assertEqual(self.request(token=token)[0],401)
    def test_server_context(self):
        self.request()
        req=self.calls[0]
        self.assertEqual((req.url,req.appid,req.identity,req.timeout_seconds),(VALVE_URL,5069970,'basketcorp-identity-v1',5))
    def test_client_cannot_choose_identity(self):
        for key,value in [('steamid',STEAM),('appid',480),('identity','other'),('user_id','u2')]:
            self.assertEqual(self.request(body={'ticket':'ab12',key:value})[0],400)
        self.assertFalse(self.calls)
    def test_malformed_inputs(self):
        for body in [{'ticket':'xyz'},{'ticket':'a'},{'ticket':4},{'ticket':'ab'*4097},{}]:
            self.assertEqual(self.request(body=body)[0],400)
        self.assertEqual(self.request(raw=b'{')[0],400)
        self.assertEqual(self.request(raw=b'x'*10001)[0],413)
    def test_invalid_expired_cancelled_ticket(self):
        # Valve may use the same rejection code: don't invent an expiry timestamp.
        for reason in ['invalid','expired','cancelled']:
            self.reply=(200,{'response':{'error':{'errorcode':101,'errordesc':reason}}})
            self.assertEqual(self.request(),(401,{'detail':'STEAM_TICKET_REJECTED'}))
        self.assertEqual(self.count(),0)
    def test_network_and_provider_config(self):
        for reply in [TimeoutError('private ticket'),OSError('private URL'),(429,{}),(503,{}),(403,{})]:
            self.reply=reply
            self.assertEqual(self.request(),(503,{'detail':'STEAM_UNAVAILABLE'}))
        self.assertEqual(self.count(),0)
    def test_malformed_valve_and_wrong_context(self):
        for reply in [(200,{}),(200,{'response':{'params':{'result':'FAIL'}}}),ok(steam='bad'),ok(steam=str(2**64)),ok(appid=480),ok(identity='other')]:
            self.reply=reply
            self.assertEqual(self.request()[0],502)
        self.assertEqual(self.count(),0)
    def test_unique_both_directions(self):
        self.assertEqual(self.request()[0],201)
        self.assertEqual(self.request('u2')[0],409)
        self.reply=ok('76561198000000002')
        self.assertEqual(self.request()[0],409)
        self.assertEqual(self.count(),1)
    def test_concurrent_conflict(self):
        barrier=Barrier(2)
        def synchronized_valve(_):
            barrier.wait(timeout=5)
            return ok()
        service=Service(self.store,synchronized_valve)
        def attempt(user):
            try:return service.link(user,{'ticket':'ab12'})[0]
            except Rejected as e:return e.status
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(attempt,['u1','u2'])),[201,409])
        self.assertEqual(self.count(),1)
    def test_concurrent_same_pair(self):
        barrier=Barrier(2)
        def synchronized_valve(_):
            barrier.wait(timeout=5)
            return ok()
        service=Service(self.store,synchronized_valve)
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses=list(pool.map(lambda _:service.link('u1',{'ticket':'ab12'})[0],range(2)))
        self.assertEqual(sorted(statuses),[200,201])
    def test_no_sensitive_logging(self):
        output=io.StringIO()
        with redirect_stdout(output),redirect_stderr(output):
            self.reply=TimeoutError('ab12 secret')
            status,body=self.request()
        self.assertEqual(output.getvalue(),'')
        self.assertNotIn('ab12',json.dumps(body))
    def test_store_unavailable(self):
        with patch.object(Store,'check',side_effect=OperationalError('private',{},Exception('private'))):
            self.assertEqual(self.request(),(503,{'detail':'IDENTITY_STORE_UNAVAILABLE'}))


    def test_disabled_by_default_no_io(self):
        from unittest.mock import Mock
        engine = Mock(side_effect=AssertionError('No DB startup'))
        transport = Mock(side_effect=AssertionError('No Valve startup'))
        with patch.dict('os.environ', {'STEAM_IDENTITY_ENABLED':''}):
            app=FastAPI()
            app.include_router(create_router(engine, auth._require_bearer_claims, transport))
            result=asyncio.run(invoke(app, self.token(), {'ticket':'ab12'}))
        self.assertEqual(result,(503,{'detail':'STEAM_IDENTITY_DISABLED'}))
        engine.assert_not_called(); transport.assert_not_called()

    def test_missing_table_never_created(self):
        with sqlite3.connect(self.path, factory=ClosingConnection) as conn:
            conn.execute('DROP TABLE steam_identity_links')
        self.assertEqual(self.request(),(503,{'detail':'IDENTITY_STORE_UNAVAILABLE'}))
        with sqlite3.connect(self.path, factory=ClosingConnection) as conn:
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='steam_identity_links'").fetchone())

    def test_account_rechecked_after_valve(self):
        def change_account(request):
            with sqlite3.connect(self.path, factory=ClosingConnection) as conn:
                conn.execute("UPDATE users SET status='disabled' WHERE user_id='u1'")
            return ok()
        service=Service(self.store,change_account)
        with self.assertRaises(Rejected) as caught:
            service.link('u1',{'ticket':'ab12'})
        self.assertEqual(caught.exception.status,403)
        self.assertEqual(self.count(),0)

    def test_concurrent_two_steamids_same_user(self):
        barrier=Barrier(2)
        def attempt(steamid):
            def synchronized_valve(_):
                barrier.wait(timeout=5)
                return ok(steamid)
            try:
                return Service(self.store,synchronized_valve).link('u1',{'ticket':'ab12'})[0]
            except Rejected as error:
                return error.status
        with ThreadPoolExecutor(max_workers=2) as pool:
            result=list(pool.map(attempt,[STEAM,'76561198000000002']))
        self.assertEqual(sorted(result),[201,409])
        self.assertEqual(self.count(),1)

    def test_database_error_no_logs_or_input_echo(self):
        output=io.StringIO()
        with redirect_stdout(output), redirect_stderr(output), patch.object(Store,'link',side_effect=OperationalError('sensitive statement',{'ticket':'ab12'},Exception('secret'))):
            self.assertEqual(self.request(),(503,{'detail':'IDENTITY_STORE_UNAVAILABLE'}))
        self.assertEqual(output.getvalue(),'')


class AdapterTests(unittest.TestCase):
    def test_exact_request_without_real_secret(self):
        from steam_identity import ValveAdapter
        from steam_identity import ValveRequest
        from urllib.parse import urlsplit, parse_qs
        class Response(io.BytesIO):
            status=200
        class Opener:
            def open(inner, req, timeout):
                url=urlsplit(req.full_url)
                self.assertEqual(url.scheme,'https')
                self.assertEqual(url.netloc,'partner.steam-api.com')
                self.assertEqual(req.method,'GET')
                self.assertEqual(parse_qs(url.query),{'key':['PUBLIC_TEST_PLACEHOLDER'],'appid':['5069970'],'identity':[IDENTITY],'ticket':['ab12']})
                self.assertEqual(timeout,5)
                return Response(json.dumps(ok()[1]).encode())
        adapter=ValveAdapter(lambda:'PUBLIC_TEST_PLACEHOLDER', Opener())
        self.assertEqual(adapter(ValveRequest(VALVE_URL,APP_ID,IDENTITY,'ab12')),ok())
    def test_errors_redacted_and_response_bounded(self):
        from steam_identity import ValveAdapter
        from steam_identity import ValveRequest
        class Response(io.BytesIO):status=200
        class Opener:
            def open(inner,*args,**kwargs):return Response(b'x'*65537)
        with self.assertRaisesRegex(RuntimeError,'^STEAM_UNAVAILABLE$'):
            ValveAdapter(lambda:'PUBLIC_TEST_PLACEHOLDER',Opener())(ValveRequest(VALVE_URL,APP_ID,IDENTITY,'ab12'))
    def test_redirect_refused(self):
        from steam_identity import NoRedirect
        self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://other.invalid'))

    def test_missing_key_no_http(self):
        from unittest.mock import Mock
        from steam_identity import ValveAdapter, ValveRequest
        opener=Mock()
        with self.assertRaisesRegex(RuntimeError,'^STEAM_UNAVAILABLE$'):
            ValveAdapter(lambda:'',opener)(ValveRequest(VALVE_URL,APP_ID,IDENTITY,'ab12'))
        opener.open.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
