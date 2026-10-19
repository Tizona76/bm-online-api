"""Integration on a private socket-only test cluster; no real database or Valve.
All fixture rows are rolled back. A4's existing FakeValve is reused.
Server functions are AST-loaded to avoid startup/configuration/credential reads.
"""
import ast, copy, os, secrets, sys, time, types, unittest, uuid
from pathlib import Path
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
from fastapi import FastAPI, HTTPException, Header
from typing import Dict, Any
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from jose import jwt, JWTError
from sqlalchemy import create_engine, text
API=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(API),str(API/'tests')]
import steam_identity as a3
import steam_wallet as a5
import steam_payments as a4
from test_steam_payments import FakeValve
from test_steam_wallet import ASGIClient
import test_steam_wallet as base
ROOT=API.parent.parent

class RollbackEngine:
    def __init__(self, conn): self.conn=conn;self.dialect=conn.dialect
    @contextmanager
    def begin(self):
        with self.conn.begin_nested(): yield self.conn
    @contextmanager
    def connect(self): yield self.conn

@unittest.skipUnless(os.environ.get("A5_TEST_CLUSTER"), "private PostgreSQL cluster required")
class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.PostgreSQLTests.setUpClass.__func__(cls)
        import psycopg
        cls.schema='purchase_'+uuid.uuid4().hex
        with psycopg.connect(**cls.kw) as conn: conn.execute('CREATE SCHEMA '+cls.schema)
        cls.db=create_engine('postgresql+psycopg://',creator=lambda:psycopg.connect(**cls.kw,
            options='-c search_path='+cls.schema+' -c lock_timeout=5000 -c statement_timeout=10000'),hide_parameters=True)
        # Read destination DDL as data; never import main or execute its startup.
        tree=ast.parse((API/'main.py').read_text())
        schema=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_auth_init_schema')
        statements=next(ast.literal_eval(n.value) for n in ast.walk(schema) if isinstance(n,ast.Assign)
                        and any(isinstance(x,ast.Name) and x.id=='stmts' for x in n.targets))
        tables=[sql for sql in statements if 'cloud_saves_v2' in sql or 'CREATE TABLE IF NOT EXISTS users (' in sql]
        fixture=base.FIXTURE_SQL.replace('CREATE TABLE users(user_id TEXT PRIMARY KEY,status TEXT,email_verified BOOLEAN);','').replace('CREATE TABLE cloud_saves_v2(user_id TEXT,profile_uuid TEXT,career_id TEXT);','')
        with cls.db.begin() as conn:
            for sql in tables: conn.exec_driver_sql(sql)
            # Isolated fixture mirrors destination migrations 004/005, not production DDL.
            conn.exec_driver_sql("ALTER TABLE cloud_saves_v2 ADD COLUMN career_id TEXT NULL")
            conn.exec_driver_sql("ALTER TABLE cloud_saves_v2 ADD CONSTRAINT cloud_saves_v2_career_id_check CHECK (career_id IS NULL OR (length(career_id) BETWEEN 1 AND 128 AND career_id=btrim(career_id)))")
            conn.exec_driver_sql("ALTER TABLE cloud_saves_v2 DROP CONSTRAINT cloud_saves_v2_uq")
            conn.exec_driver_sql("ALTER TABLE cloud_saves_v2 ADD CONSTRAINT cloud_saves_v2_uq UNIQUE NULLS NOT DISTINCT (user_id,profile_uuid,career_id)")
            for sql in fixture.split(';'):
                if sql.strip(): conn.execute(text(sql))
            ddl=(API/'sql/steam_payments_a4.sql').read_text().split('-- BEGIN TABLE DDL')[1].split('\n',1)[1].split('-- END TABLE DDL')[0]
            conn.exec_driver_sql(ddl)
            conn.execute(text('ALTER TABLE steam_wallet_deliveries DROP CONSTRAINT steam_wallet_deliveries_receipt_id_fkey'))
            conn.execute(text('ALTER TABLE steam_wallet_deliveries ADD FOREIGN KEY(receipt_id) REFERENCES steam_receipts(receipt_id)'))
    @classmethod
    def tearDownClass(cls):
        import psycopg
        cls.db.dispose()
        with psycopg.connect(**cls.kw) as conn: conn.execute('DROP SCHEMA '+cls.schema+' CASCADE')
    def setUp(self):
        self.conn=self.db.connect();self.outer=self.conn.begin();self.engine=RollbackEngine(self.conn)
        self.addCleanup(self.conn.close);self.addCleanup(self.outer.rollback)
        self.sid=str(80000000000000000+secrets.randbelow(9999999999999999))
        self.sid2=str(int(self.sid)+1)
        self.app=FastAPI();self.fake=FakeValve()
        def valve_ticket(_adapter,req):
            assert req.appid==5069970 and req.identity==a3.IDENTITY
            if req.ticket=='00': return 200,{'response':{'error':{'errorcode':101}}}
            return 200,{'response':{'params':{'result':'OK','steamid':self.sid if req.ticket=='aa' else self.sid2}}}
        self.addCleanup(patch.stopall)
        patch.object(a3.ValveAdapter,'__call__',valve_ticket).start()
        patch.object(a4.Valve,'call',side_effect=AssertionError('REAL VALVE FORBIDDEN')).start()
        # Outer rollback transaction must never be committed by the session lock.
        patch.object(a4.Payments,'exclusive',lambda *_:nullcontext()).start()
        patch.dict(os.environ,{'STEAM_IDENTITY_ENABLED':'1','STEAM_PAYMENTS_ENABLED':'1','STEAM_PAYMENTS_ENVIRONMENT':'sandbox','STEAM_PAYMENTS_LIVE_ENABLED':'0'}).start()
        source=(API/'main.py').read_text()
        names={'_jwt_verify_access','_require_bearer_claims','_steam_require_purchase_bearer','_steam_purchase_session','steam_purchase_session','SteamPurchaseSessionPayload'}
        nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name in names]
        self.ns=dict(jwt=jwt,JWTError=JWTError,JWT_SECRET=secrets.token_hex(32),HTTPException=HTTPException,
                     _steam_identity=a3,_lb_get_engine=lambda:self.engine,JWT_ISSUER="integration-test",JWT_AUDIENCE="general-test",Dict=Dict,Any=Any,Header=Header,time=time,os=os,BaseModel=BaseModel,app=self.app,
                     JSONResponse=JSONResponse,run_in_threadpool=run_in_threadpool)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<destination purchase functions>','exec'),self.ns)
        auth=self.ns['_steam_require_purchase_bearer']
        self.app.include_router(a4.create_router(lambda:self.engine,auth,valve=self.fake,enabled=lambda:True,purchase_sessions=True))
        self.app.include_router(a5.create_router(lambda:self.engine,auth,enabled=lambda:True,purchase_sessions=True))
        self.client=ASGIClient(self.app)
        self.career='test_'+uuid.uuid4().hex
    def session(self,ticket='aa'):
        r=self.client.request('POST','/v1/steam/purchase-session',json={'ticket':ticket})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()
    def request(self,method,path,session,body=None,key='testkey'):
        return self.client.request(method,path,json=body,headers={'Authorization':'Bearer '+session['access_token'],'Idempotency-Key':key})
    def enroll(self,s):
        body={'profile_uuid':'test_profile','career_id':self.career,'snapshot':{'profile_id':'local1','career_id':self.career,'wallet':{'tokens':17},'season_number':3},'accept_initial_registration':True}
        r=self.request('POST','/v1/steam/wallet/enroll',s,body)
        self.assertIn(r.status_code,(200,201),r.text)
        return body,r.json()
    def test_destination_save_id_and_retry_preserved(self):
        session=self.session();body,checkpoint=self.enroll(session)
        row=self.conn.execute(text('SELECT save_id,rev,blob_json FROM cloud_saves_v2 WHERE career_id=:c'),{'c':self.career}).one()
        self.assertEqual(len(row.save_id),32);self.assertEqual(row.rev,1)
        self.assertEqual(self.request('POST','/v1/steam/wallet/enroll',session,body).json(),checkpoint)
        self.assertEqual(self.conn.execute(text('SELECT save_id FROM cloud_saves_v2 WHERE career_id=:c'),{'c':self.career}).scalar(),row.save_id)
    def test_disabled_session_no_identity_created(self):
        before=self.conn.execute(text('SELECT count(*) FROM users')).scalar()
        with patch.dict(os.environ,{'STEAM_IDENTITY_ENABLED':'0'}):
            response=self.client.request('POST','/v1/steam/purchase-session',json={'ticket':'aa'})
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.json()['detail'],'STEAM_IDENTITY_DISABLED')
        self.assertEqual(self.conn.execute(text('SELECT count(*) FROM users')).scalar(),before)
    def test_general_jwt_fallback_and_purchase_scope(self):
        session=self.session();now=int(time.time())
        claims={'sub':session['user_id'],'typ':'access','iss':self.ns['JWT_ISSUER'],'aud':self.ns['JWT_AUDIENCE'],'iat':now,'exp':now+60}
        access=jwt.encode(claims,self.ns['JWT_SECRET'],algorithm='HS256')
        self.assertEqual(self.ns['_steam_require_purchase_bearer']('Bearer '+access)['typ'],'access')
        purchase=jwt.decode(session['access_token'],self.ns['JWT_SECRET'],algorithms=['HS256'],audience='steam-purchases',issuer='integration-test/steam-purchases')
        for edit in [{'typ':'access'},{'app_id':1},{'service_identity':'wrong'},{'exp':now+901},{'aud':'general-test'}]:
            altered=jwt.encode(purchase|edit,self.ns['JWT_SECRET'],algorithm='HS256')
            with self.assertRaises(HTTPException): self.ns['_steam_require_purchase_bearer']('Bearer '+altered)
    def test_identity_no_email_reused_and_scoped(self):
        s=self.session();self.assertEqual(s['user_id'],self.session()['user_id'])
        row=self.conn.execute(text('SELECT email_verified,status,email FROM users WHERE user_id=:u'),{'u':s['user_id']}).one()
        self.assertEqual(row[0:2],(False,'active'));self.assertTrue(row[2].endswith('@identity.invalid'))
        with self.assertRaises(HTTPException):self.ns['_require_bearer_claims']('Bearer '+s['access_token'])
        with self.assertRaises(a3.Rejected):a3.Store(lambda:self.engine).check(s['user_id'])
        self.assertEqual(self.request('GET','/v1/steam/payments/catalog',s).status_code,200)
        claims=self.ns['_steam_require_purchase_bearer']('Bearer '+s['access_token']);claims['exp']=int(time.time())-1
        expired=dict(s,access_token=jwt.encode(claims,self.ns['JWT_SECRET'],algorithm='HS256'))
        self.assertEqual(self.request('GET','/v1/steam/payments/catalog',expired).status_code,401)
    def test_invalid_ticket_creates_nothing_and_disabled_never_replaced(self):
        before=self.conn.execute(text('SELECT count(*) FROM users')).scalar()
        self.assertEqual(self.client.request('POST','/v1/steam/purchase-session',json={'ticket':'00'}).status_code,401)
        self.assertEqual(self.conn.execute(text('SELECT count(*) FROM users')).scalar(),before)
        s=self.session();self.conn.execute(text("UPDATE users SET status='disabled' WHERE user_id=:u"),{'u':s['user_id']})
        self.assertEqual(self.client.request('POST','/v1/steam/purchase-session',json={'ticket':'aa'}).status_code,403)
        self.assertEqual(self.conn.execute(text('SELECT count(*) FROM users')).scalar(),before+1)
    def test_existing_verified_link_preserved(self):
        uid='test_'+uuid.uuid4().hex
        self.conn.execute(text("INSERT INTO users(user_id,status,email_verified,email) VALUES(:u,'active',TRUE,:e)"),{'u':uid,'e':uid+'@example.invalid'})
        self.conn.execute(text('INSERT INTO steam_identity_links(user_id,steam_id,app_id,service_identity) VALUES(:u,:s,5069970,:i)'),{'u':uid,'s':self.sid,'i':a3.IDENTITY})
        self.assertEqual(self.session()['user_id'],uid)
    def test_enroll_conflict_repeat_and_balance(self):
        s=self.session();body,cp=self.enroll(s)
        self.assertEqual(cp['snapshot']['wallet']['tokens'],17)
        self.assertEqual(self.request('POST','/v1/steam/wallet/enroll',s,body).json(),cp)
        other=self.session('bb');bad=copy.deepcopy(body);bad['profile_uuid']='changed_profile'
        self.assertEqual(self.request('POST','/v1/steam/wallet/enroll',other,bad).status_code,409)
        body['snapshot']['wallet']['tokens']=2
        self.assertEqual(self.request('POST','/v1/steam/wallet/enroll',s,body).status_code,409)
        self.assertEqual(len(self.fake.calls),0)
    def test_registration_locks_block_competing_writers(self):
        from sqlalchemy.exc import OperationalError
        s=self.session();self.enroll(s)
        with self.db.connect() as other:
            with other.begin():
                other.execute(text("SET LOCAL lock_timeout='100ms'"))
                with self.assertRaises(OperationalError):
                    with other.begin_nested():
                        other.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:sid,0))'),{'sid':'steam_identity:'+self.sid})
                with self.assertRaises(OperationalError):
                    with other.begin_nested():
                        other.execute(text('LOCK TABLE cloud_saves_v2 IN ROW EXCLUSIVE MODE'))
    def test_existing_cloud_profile_reused_without_migration(self):
        s=self.session()
        self.conn.execute(text('INSERT INTO cloud_saves_v2(save_id,user_id,profile_uuid,career_id) VALUES(:sid,:u,:p,:c)'),{'sid':uuid.uuid4().hex,'u':s['user_id'],'p':'old_cloud_profile','c':self.career})
        _,cp=self.enroll(s)
        self.assertEqual(cp['profile_uuid'],'old_cloud_profile')
        self.assertEqual(cp['snapshot']['wallet']['tokens'],17)
    def test_no_checkpoint_no_order_and_session_renewal(self):
        s=self.session();body={'profile_uuid':'test_profile','career_id':self.career,'pack_id':'tokens75','language':'fr'}
        r=self.request('POST','/v1/steam/payments/orders',s,body)
        self.assertEqual(r.status_code,409);self.assertEqual(r.json()['detail'],'CHECKPOINT_REQUIRED');self.assertEqual(self.fake.calls,[])
        self.enroll(s);r=self.request('POST','/v1/steam/payments/orders',s,body)
        self.assertEqual(r.status_code,201,r.text);oid=r.json()['order_id']
        renewed=self.session();repeat=self.request('POST','/v1/steam/payments/orders',renewed,body)
        self.assertEqual(repeat.json()['order_id'],oid);self.assertEqual(self.fake.count('InitTxn'),1)
        self.assertEqual(r.json()['pack_id'],'tokens75')
    def test_simulated_confirm_delivery_once_and_cancel(self):
        for pack,amount in [('tokens25',25),('tokens75',75),('tokens150',150)]:
            with self.subTest(pack=pack):
                self.career='test_'+uuid.uuid4().hex
                s=self.session();body,cp=self.enroll(s)
                order={'profile_uuid':body['profile_uuid'],'career_id':self.career,'pack_id':pack,'language':'fr'}
                created=self.request('POST','/v1/steam/payments/orders',s,order,key=self.career)
                self.assertEqual(created.status_code,201,created.text);oid=created.json()['order_id']
                self.fake.set(oid,'Approved')
                confirmed=self.request('POST','/v1/steam/payments/orders/'+oid+'/authorization',s,{'authorized':True})
                self.assertEqual(confirmed.json()['state'],'PAID',confirmed.text)
                again=self.request('POST','/v1/steam/payments/orders/'+oid+'/resume',self.session(),{})
                self.assertEqual(confirmed.json()['receipt'],again.json()['receipt'])
                delivery={'receipt_id':confirmed.json()['receipt']['receipt_id'],'profile_uuid':body['profile_uuid'],'career_id':self.career,'expected_revision':cp['revision'],'parent_hash':cp['checkpoint_hash'],'snapshot':cp['snapshot']}
                r=self.request('POST','/v1/steam/wallet/deliveries',s,delivery,key=self.career)
                self.assertEqual(r.status_code,201,r.text);d=r.json()
                self.assertEqual(d['candidate']['wallet']['tokens'],17+amount)
                repeated=self.request('POST','/v1/steam/wallet/deliveries',s,delivery,key=self.career)
                self.assertEqual(repeated.json()['operation_id'],d['operation_id'])
                ack={'candidate_hash':d['candidate_hash']}
                acknowledged=self.request('POST','/v1/steam/wallet/deliveries/'+d['operation_id']+'/ack',s,ack)
                self.assertEqual(acknowledged.status_code,200,acknowledged.text)
                self.assertEqual(self.request('POST','/v1/steam/wallet/deliveries/'+d['operation_id']+'/ack',self.session(),ack).json(),acknowledged.json())
                cp2=self.request('GET','/v1/steam/wallet/checkpoint?profile_uuid=test_profile&career_id='+self.career,s).json()
                self.assertEqual(cp2['snapshot']['wallet']['tokens'],17+amount)
        self.career='test_'+uuid.uuid4().hex;s=self.session();body,cp=self.enroll(s)
        order={'profile_uuid':'test_profile','career_id':self.career,'pack_id':'tokens25','language':'fr'}
        r=self.request('POST','/v1/steam/payments/orders',s,order,key=self.career);oid=r.json()['order_id']
        cancelled=self.request('POST','/v1/steam/payments/orders/'+oid+'/authorization',s,{'authorized':False})
        self.assertEqual(cancelled.json()['state'],'CANCELLED');self.assertIsNone(cancelled.json()['receipt'])

if __name__=='__main__':unittest.main(verbosity=2)
