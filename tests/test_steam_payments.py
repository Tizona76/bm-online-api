"""A4 synthetic API/PG tests. No main import, real Valve call or real credentials.
Reuse A5's strict socket-only PostgreSQL 18.6 fixture isolation checks.
A5_TEST_CLUSTER must be a private, task-created /private/tmp cluster.
"""
import copy
import importlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from unittest.mock import patch

from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import steam_payments as pay
import steam_wallet as sw
import test_steam_wallet as base


class FakeValve:
    def __init__(self):
        self.orders={};self.calls=[];self.timeout_init=False;self.timeout_final=False
        self.timeout_query=False;self.final_no_effect=False;self.currency='EUR';self.report=[]
        self.query_edit=None
    def set(self,oid,state):
        self.orders[oid]['status']=state;self.orders[oid]['items'][0]['itemstatus']=state
    def call(self,method,env,params):
        assert env=='sandbox' and params['appid']==5069970
        self.calls.append((method,copy.deepcopy(params)))
        oid=params.get('orderid')
        if method=='GetUserInfo': p={'currency':self.currency,'status':'Active'}
        elif method=='InitTxn':
            assert oid not in self.orders
            self.orders[oid]={'orderid':oid,'transid':str(int(oid)+2**63),'steamid':params['steamid'],
                'currency':'EUR','status':'Init','items':[{'itemid':params['itemid[0]'],'qty':1,
                'amount':params['amount[0]']-16,'vat':16,'itemstatus':'Init'}]}
            if self.timeout_init:raise TimeoutError('synthetic-secret')
            p={'orderid':oid,'transid':self.orders[oid]['transid']}
        elif method=='QueryTxn':
            if self.timeout_query:raise TimeoutError('synthetic-secret')
            if oid not in self.orders:return {'response':{'result':'Failure','error':{'errorcode':3}}}
            p=copy.deepcopy(self.orders[oid])
            if self.query_edit:self.query_edit(p)
        elif method=='FinalizeTxn':
            assert self.orders[oid]['status']=='Approved'
            if not self.final_no_effect:self.set(oid,'Succeeded')
            if self.timeout_final:raise TimeoutError('synthetic-secret')
            p={'orderid':oid,'transid':self.orders[oid]['transid']}
        elif method=='GetReport':p={'count':len(self.report),'orders':copy.deepcopy(self.report)}
        else:raise AssertionError(method)
        return {'response':{'result':'OK','params':p}}
    def count(self,method):return sum(m==method for m,_ in self.calls)


def client(engine=lambda:None,valve=None,enabled=None):
    app=FastAPI();app.include_router(pay.create_router(engine,base.auth._require_bearer_claims,valve=valve,enabled=enabled))
    @app.get('/existing')
    def existing():return {'unchanged':True}
    return base.ASGIClient(app)


class RouteTests(unittest.TestCase):
    def test_disabled_every_route(self):
        with patch.dict(os.environ,{},clear=True):
            c=client(lambda:self.fail('database touched'))
            for method,path in [('GET','catalog'),('POST','orders'),('GET','orders/1'),('POST','orders/1/authorization'),('POST','orders/1/resume'),('GET','receipts')]:
                response=c.request(method,'/v1/steam/payments/'+path)
                self.assertEqual(response.status_code,503);self.assertEqual(response.json()['detail'],'STEAM_PAYMENTS_DISABLED')
    def test_auth_missing_invalid_expired(self):
        for token in ('','invalid',base.token(exp=0)):
            self.assertEqual(client(enabled=lambda:True).get('/v1/steam/payments/catalog',headers={'Authorization':'Bearer '+token}).status_code,401)
    def test_import_no_io(self):
        with patch('socket.socket.connect',side_effect=AssertionError('network')),patch('sqlalchemy.create_engine',side_effect=AssertionError('db')):
            importlib.reload(pay);client(lambda:self.fail('provider'))
    def test_existing_route(self):self.assertEqual(client().get('/existing').json(),{'unchanged':True})
    def test_live_double_gate(self):
        with patch.dict(os.environ,{'STEAM_PAYMENTS_ENVIRONMENT':'live'},clear=True):
            with self.assertRaises(sw.Rejected):pay.environment()
    def test_transport_redacted(self):
        class Opener:
            def open(self,*a,**k):raise RuntimeError('SECRET_TEST_KEY')
        stream=io.StringIO()
        with redirect_stdout(stream),redirect_stderr(stream):
            with self.assertRaises(sw.Rejected) as error:pay.Valve(lambda:'SECRET_TEST_KEY',Opener()).call('QueryTxn','sandbox',{'appid':5069970,'orderid':'1'})
        self.assertNotIn('SECRET_TEST_KEY',str(error.exception));self.assertEqual(stream.getvalue(),'')
    def test_transport_versions_params(self):
        class Reply:
            status=200
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self,n):return b'{"response":{"result":"OK","params":{}}}'
        class Opener:
            def open(self,request,timeout):self.request=request;self.timeout=timeout;return Reply()
        from urllib.parse import urlsplit,parse_qs
        for method,(verb,version) in pay.VERSIONS.items():
            opener=Opener();pay.Valve(lambda:'fake-key',opener).call(method,'sandbox',{'appid':5069970})
            request=opener.request;self.assertEqual(request.method,verb)
            self.assertIn('/ISteamMicroTxnSandbox/'+method+'/v'+str(version)+'/',request.full_url)
            body=request.data.decode() if request.data else urlsplit(request.full_url).query
            self.assertEqual(parse_qs(body)['appid'],['5069970']);self.assertEqual(opener.timeout,5)
    def test_no_a4_source_enabled_by_default(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(sw.Rejected) as e:sw.A4Receipts().lock(None,'receipt')
            self.assertEqual(e.exception.code,'STEAM_A4_UNAVAILABLE')


@unittest.skipUnless(os.environ.get('A5_TEST_CLUSTER'),'isolated PostgreSQL required')
class PostgreSQLTests(unittest.TestCase):
    setUpClass=classmethod(base.PostgreSQLTests.setUpClass.__func__)
    def setUp(self):
        import psycopg
        self.schema='a4_'+uuid.uuid4().hex
        with psycopg.connect(**self.kw) as c:c.execute('CREATE SCHEMA '+self.schema)
        self.engine=create_engine('postgresql+psycopg://',creator=lambda:psycopg.connect(**self.kw,options='-c search_path='+self.schema+' -c lock_timeout=5000 -c statement_timeout=10000'),hide_parameters=True)
        with self.engine.begin() as conn:
            for sql in base.FIXTURE_SQL.split(';'):
                if sql.strip():conn.execute(text(sql))
            conn.execute(text("INSERT INTO users VALUES ('u1','active',TRUE),('u2','active',TRUE),('guest','active',FALSE)"))
            conn.execute(text("INSERT INTO steam_identity_links VALUES ('u1','76561198000000001',5069970,'basketcorp-identity-v1')"))
            conn.execute(text("INSERT INTO cloud_saves_v2 VALUES ('u1','profile1','career1')"))
            ddl=(Path(__file__).resolve().parents[1]/'sql/steam_payments_a4.sql').read_text().split('-- BEGIN TABLE DDL')[1].split('\n',1)[1].split('-- END TABLE DDL')[0]
            conn.exec_driver_sql(ddl)
            conn.execute(text('ALTER TABLE steam_wallet_deliveries DROP CONSTRAINT steam_wallet_deliveries_receipt_id_fkey'))
            conn.execute(text('ALTER TABLE steam_wallet_deliveries ADD FOREIGN KEY(receipt_id) REFERENCES steam_receipts(receipt_id)'))
        self.valve=FakeValve();self.s=pay.Payments(lambda:self.engine,self.valve)
        self.c=client(lambda:self.engine,self.valve,lambda:True)
        self.wallet=sw.Wallet(lambda:self.engine,pay.ReceiptSource())
        _,self.cp=self.wallet.put_checkpoint('u1',{'profile_uuid':'profile1','career_id':'career1','expected_revision':0,'parent_hash':None,
            'snapshot':{'career_id':'career1','wallet':{'tokens':10},'season_tokens_earned':7}},'initial')
        self.body={'pack_id':'tokens25','profile_uuid':'profile1','career_id':'career1','language':'fr'}
    def tearDown(self):
        import psycopg
        self.engine.dispose()
        with psycopg.connect(**self.kw) as conn:conn.execute('DROP SCHEMA '+self.schema+' CASCADE')
    def req(self,method,path,body=None,key='key1',user='u1'):
        return self.c.request(method,'/v1/steam/payments/'+path,json=body,headers={'Authorization':'Bearer '+base.token(user),'Idempotency-Key':key})
    def create(self,**body):
        response=self.req('POST','orders',self.body|body);self.assertIn(response.status_code,(201,202),response.text);return response.json()['order_id']
    def confirm(self,oid):
        self.valve.set(oid,'Approved');return self.req('POST','orders/'+oid+'/authorization',{'authorized':True})
    def count(self,table):
        with self.engine.connect() as c:return c.execute(text('SELECT count(*) FROM '+table)).scalar()
    def test_catalog(self):
        r=self.req('GET','catalog');self.assertEqual(r.status_code,200)
        self.assertEqual([(p['tokens'],p['amount_minor']) for p in r.json()['packs']],[(25,99),(75,299),(150,499)])
    def test_missing_a3(self):self.assertEqual(self.req('POST','orders',self.body,user='u2').status_code,409)
    def test_guest(self):self.assertEqual(self.req('POST','orders',self.body,user='guest').status_code,403)
    def test_client_price_identity_rejected(self):
        for key in ('steam_id','quantity','price','environment','authorized'):
            self.assertEqual(self.req('POST','orders',self.body|{key:'fake'}).status_code,400)
        self.assertEqual(self.valve.count('InitTxn'),0)
    def test_wrong_destination(self):self.assertEqual(self.req('POST','orders',self.body|{'career_id':'other'}).status_code,409)
    def test_create_exact_price(self):
        self.create();p=next(p for m,p in self.valve.calls if m=='InitTxn')
        self.assertEqual((p['amount[0]'],p['qty[0]'],p['itemid[0]'],p['currency']),(99,1,1,'EUR'));self.assertEqual(self.count('steam_receipts'),0)
    def test_non_eur_uses_eur_order(self):
        self.valve.currency='USD';oid=self.create()
        p=next(p for m,p in self.valve.calls if m=='InitTxn')
        self.assertEqual((p['currency'],p['amount[0]'],p['qty[0]'],p['itemid[0]']),('EUR',99,1,1))
        self.assertEqual(self.confirm(oid).json()['state'],'PAID')
    def test_report_purchase_identity_without_verified_email(self):
        oid=self.create();self.confirm(oid);self.valve.set(oid,'Refunded')
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE users SET email_verified=FALSE WHERE user_id='u1'"))
        self.valve.report=[dict(self.valve.orders[oid],time='2026-10-19T10:00:00Z')]
        calls=(self.valve.count('InitTxn'),self.valve.count('FinalizeTxn'))
        self.s.reconcile_report('2026-10-19T00:00:00Z')
        self.s.reconcile_report('2026-10-19T00:00:00Z')
        self.assertEqual((self.valve.count('InitTxn'),self.valve.count('FinalizeTxn')),calls)
        self.assertEqual(self.count('steam_receipts'),1)
        with self.engine.connect() as conn:
            self.assertEqual(conn.execute(text('SELECT state FROM steam_receipts')).scalar(),'revoked')
        self.assertIsNone(self.s.wallet.steam_id)
    def test_same_key(self):
        oid=self.create();r=self.req('POST','orders',self.body);self.assertEqual(r.json()['order_id'],oid);self.assertEqual(self.valve.count('InitTxn'),1)
    def test_changed_key_payload(self):
        self.create();self.assertEqual(self.req('POST','orders',self.body|{'pack_id':'tokens75'}).status_code,409)
    def test_other_key_blocked_while_uncertain(self):
        self.create();self.assertEqual(self.req('POST','orders',self.body,key='new').status_code,409)
    def test_unapproved_callback_cannot_capture(self):
        oid=self.create();r=self.req('POST','orders/'+oid+'/authorization',{'authorized':True})
        self.assertEqual(r.json()['state'],'AWAITING');self.assertEqual(self.valve.count('FinalizeTxn'),0);self.assertEqual(self.count('steam_receipts'),0)
    def test_cancel_and_resume_no_capture(self):
        oid=self.create();self.valve.set(oid,'Approved');r=self.req('POST','orders/'+oid+'/authorization',{'authorized':False})
        self.assertEqual(r.json()['state'],'CANCELLED');self.req('POST','orders/'+oid+'/resume',{});self.assertEqual(self.valve.count('FinalizeTxn'),0)
    def test_denied(self):
        oid=self.create();self.valve.set(oid,'Failed');r=self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(r.json()['state'],'FAILED');self.assertEqual(self.count('steam_receipts'),0)
    def test_confirmation_and_repeat(self):
        oid=self.create();r=self.confirm(oid);self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.json()['state'],'PAID')
        self.req('POST','orders/'+oid+'/resume',{});self.assertEqual(self.valve.count('FinalizeTxn'),1);self.assertEqual(self.count('steam_receipts'),1)
    def test_timeout_init_resume(self):
        self.valve.timeout_init=True;oid=self.create();self.assertEqual(self.s.get('u1',oid)[1]['state'],'INIT_UNKNOWN')
        self.confirm(oid);self.assertEqual(self.valve.count('InitTxn'),1);self.assertEqual(self.count('steam_receipts'),1)
    def test_timeout_final_after_capture(self):
        self.valve.timeout_final=True;oid=self.create();self.confirm(oid);self.assertEqual(self.count('steam_receipts'),1);self.assertEqual(self.valve.count('FinalizeTxn'),1)
    def test_timeout_final_before_capture_never_recaptures(self):
        self.valve.final_no_effect=True;self.valve.timeout_final=True;oid=self.create();self.confirm(oid)
        self.req('POST','orders/'+oid+'/resume',{});self.assertEqual(self.valve.count('FinalizeTxn'),1);self.assertEqual(self.count('steam_receipts'),0)
        self.assertEqual(self.s.get('u1',oid)[1]['state'],'FINALIZE_UNKNOWN')
    def test_query_timeout(self):
        oid=self.create();self.valve.timeout_query=True;r=self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(r.status_code,202);self.assertNotIn('synthetic-secret',r.text);self.assertEqual(self.count('steam_receipts'),0)
    def test_context_and_tax_validation(self):
        oid=self.create();self.valve.set(oid,'Succeeded')
        edits=[lambda p:p.update(steamid='76561198000000099'),lambda p:p.update(currency='USD'),lambda p:p.update(orderid='8'),
               lambda p:p['items'][0].update(vat=999),lambda p:p['items'][0].update(qty=2),lambda p:p['items'][0].update(itemid=9)]
        for edit in edits:
            self.valve.query_edit=edit;self.assertEqual(self.req('POST','orders/'+oid+'/resume',{}).status_code,502)
        self.assertEqual(self.count('steam_receipts'),0)
    def test_incomplete_item_no_receipt(self):
        oid=self.create();self.valve.set(oid,'Succeeded');self.valve.orders[oid]['items'][0]['itemstatus']='Init'
        self.req('POST','orders/'+oid+'/resume',{});self.assertEqual(self.count('steam_receipts'),0)
    def test_concurrent_creation(self):
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda _:self.req('POST','orders',self.body),range(4)))
        self.assertTrue(all(r.status_code in (200,201) for r in rows));self.assertEqual(self.count('steam_orders'),1);self.assertEqual(self.valve.count('InitTxn'),1)
    def test_concurrent_finalization(self):
        oid=self.create();self.valve.set(oid,'Approved')
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda _:self.req('POST','orders/'+oid+'/authorization',{'authorized':True}),range(4)))
        self.assertTrue(all(r.status_code in (200,409) for r in rows));self.assertEqual(self.valve.count('FinalizeTxn'),1);self.assertEqual(self.count('steam_receipts'),1)
    def test_claim_without_advisory_lock(self):
        oid=self.create()
        with ThreadPoolExecutor(max_workers=4) as pool:claims=list(pool.map(lambda _:self.s.mark('u1',oid,'FINALIZE_UNKNOWN',expected=('AWAITING',)),range(4)))
        self.assertEqual(claims.count(True),1)
    def test_receipt_A5(self):
        oid=self.create();r=self.confirm(oid);rid=r.json()['receipt']['receipt_id']
        payload={'receipt_id':rid,'profile_uuid':'profile1','career_id':'career1','expected_revision':self.cp['revision'],
                 'parent_hash':self.cp['checkpoint_hash'],'snapshot':copy.deepcopy(self.cp['snapshot'])}
        with patch.dict(os.environ,{'STEAM_PAYMENTS_ENABLED':'1','STEAM_PAYMENTS_ENVIRONMENT':'sandbox'}):
            real=sw.Wallet(lambda:self.engine);_,d=real.create_delivery('u1',payload,'d1')
        self.assertEqual(d['candidate']['wallet']['tokens'],35)
        self.assertEqual(self.wallet.get_checkpoint('u1',{'profile_uuid':'profile1','career_id':'career1'})[1]['snapshot']['wallet']['tokens'],10)
        self.assertEqual(d['candidate']['season_tokens_earned'],7)
    def test_refund_unapplied(self):
        oid=self.create();self.confirm(oid);self.valve.set(oid,'Refunded');r=self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(r.json()['receipt']['state'],'revoked');self.valve.set(oid,'Succeeded');self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(self.s.get('u1',oid)[1]['receipt']['state'],'revoked')
    def test_refund_reserved_manual_no_debit(self):
        oid=self.create();rid=self.confirm(oid).json()['receipt']['receipt_id']
        _,d=self.wallet.create_delivery('u1',{'receipt_id':rid,'profile_uuid':'profile1','career_id':'career1',
            'expected_revision':1,'parent_hash':self.cp['checkpoint_hash'],'snapshot':self.cp['snapshot']},'d1')
        self.wallet.acknowledge('u1',d['operation_id'],{'candidate_hash':d['candidate_hash']})
        self.valve.set(oid,'Refunded');r=self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(r.json()['receipt']['state'],'manual')
        self.assertEqual(self.wallet.get_checkpoint('u1',{'profile_uuid':'profile1','career_id':'career1'})[1]['snapshot']['wallet']['tokens'],35)
    def test_partial_refund_manual(self):
        oid=self.create();self.confirm(oid);self.valve.set(oid,'PartialRefund');r=self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(r.json()['receipt']['state'],'manual')
    def test_get_never_finalizes(self):
        oid=self.create();self.valve.set(oid,'Approved');self.req('GET','orders/'+oid);self.assertEqual(self.valve.count('QueryTxn'),0);self.assertEqual(self.valve.count('FinalizeTxn'),0)
    def test_unknown_and_foreign(self):
        oid=self.create();self.assertEqual(self.req('GET','orders/1').status_code,404)
        with self.engine.begin() as c:c.execute(text("INSERT INTO steam_identity_links VALUES ('u2','76561198000000002',5069970,'basketcorp-identity-v1')"))
        self.assertEqual(self.req('GET','orders/'+oid,user='u2').status_code,404)
    def test_receipt_cursor_not_random_orderid(self):
        with patch('steam_payments.secrets.randbelow',return_value=899):oid=self.create()
        self.confirm(oid);first=self.req('GET','receipts?profile_uuid=profile1&career_id=career1&cursor=0').json()
        with patch('steam_payments.secrets.randbelow',return_value=99):
            r=self.req('POST','orders',self.body,key='key2');oid=r.json()['order_id']
        self.confirm(oid);second=self.req('GET','receipts?profile_uuid=profile1&career_id=career1&cursor='+first['next_cursor']).json()
        self.assertEqual(len(second['receipts']),1);self.assertEqual(second['receipts'][0]['order_id'],'100')
    def test_report_requery_refund_and_cursor(self):
        oid=self.create();self.confirm(oid);self.valve.set(oid,'Refunded')
        self.valve.report=[dict(self.valve.orders[oid],time='2026-10-16T10:00:00Z')]
        r=self.s.reconcile_report('2026-10-16T00:00:00Z');self.assertFalse(r['complete']);self.assertEqual(r['next_window'],'2026-10-16T10:00:00Z')
        self.assertEqual(self.s.get('u1',oid)[1]['receipt']['state'],'revoked');self.s.reconcile_report(r['next_window'])
        self.assertEqual(self.count('steam_receipts'),1)
    def test_sql_immutable_and_unconfirmed(self):
        oid=self.create()
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as c:c.execute(text('UPDATE steam_orders SET amount_minor=1 WHERE order_id=:id'),{'id':int(oid)})
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as c:c.execute(text("INSERT INTO steam_receipts(receipt_id,order_id,state) VALUES('fake',:id,'confirmed')"),{'id':int(oid)})
        self.confirm(oid)
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as c:c.execute(text('DELETE FROM steam_receipts'))
    def test_terminal_state_cannot_be_overwritten_by_stale_worker(self):
        oid=self.create();self.confirm(oid)
        self.assertFalse(self.s.mark('u1',oid,'FAILED',expected=('NEW',)))
        self.assertFalse(self.s.mark('u1',oid,'CANCELLED',expected=('INIT_UNKNOWN','AWAITING')))
        self.assertEqual(self.s.get('u1',oid)[1]['state'],'PAID')
    def test_durable_final_claim_restart(self):
        oid=self.create();self.valve.set(oid,'Approved')
        self.assertTrue(self.s.mark('u1',oid,'FINALIZE_UNKNOWN',expected=('AWAITING',)))
        restarted=pay.Payments(lambda:self.engine,self.valve)
        restarted.reconcile('u1',oid,True)
        self.assertEqual(self.valve.count('FinalizeTxn'),0);self.assertEqual(self.count('steam_receipts'),0)
        self.valve.set(oid,'Succeeded');restarted.reconcile('u1',oid,True)
        self.assertEqual(self.count('steam_receipts'),1)
    def test_missing_valve_order_never_reinitializes(self):
        self.valve.timeout_init=True;oid=self.create();del self.valve.orders[oid]
        self.req('POST','orders/'+oid+'/resume',{})
        self.assertEqual(self.valve.count('InitTxn'),1);self.assertEqual(self.count('steam_receipts'),0)
    def test_refund_blocks_pending_A5_ack(self):
        oid=self.create();rid=self.confirm(oid).json()['receipt']['receipt_id']
        _,d=self.wallet.create_delivery('u1',{'receipt_id':rid,'profile_uuid':'profile1','career_id':'career1',
            'expected_revision':1,'parent_hash':self.cp['checkpoint_hash'],'snapshot':self.cp['snapshot']},'d1')
        self.valve.set(oid,'Chargedback');self.s.reconcile('u1',oid)
        with self.assertRaises(sw.Rejected):self.wallet.acknowledge('u1',d['operation_id'],{'candidate_hash':d['candidate_hash']})
        self.assertEqual(self.wallet.get_checkpoint('u1',{'profile_uuid':'profile1','career_id':'career1'})[1]['snapshot']['wallet']['tokens'],10)
    def test_before_confirmation_A5_has_no_receipt(self):
        oid=self.create()
        with self.assertRaises(sw.Rejected) as error:self.wallet.create_delivery('u1',{'receipt_id':'steam-sandbox-'+oid,'profile_uuid':'profile1','career_id':'career1',
            'expected_revision':1,'parent_hash':self.cp['checkpoint_hash'],'snapshot':self.cp['snapshot']},'d1')
        self.assertEqual(error.exception.code,'RECEIPT_NOT_FOUND');self.assertEqual(self.count('steam_wallet_deliveries'),0)
    def test_environment_isolation(self):
        oid=self.create();rid=self.confirm(oid).json()['receipt']['receipt_id']
        with patch.dict(os.environ,{'STEAM_PAYMENTS_ENABLED':'1','STEAM_PAYMENTS_ENVIRONMENT':'live','STEAM_PAYMENTS_LIVE_ENABLED':'1'}):
            with self.engine.begin() as c:
                with self.assertRaises(sw.Rejected) as error:sw.A4Receipts().lock(c,rid)
                self.assertEqual(error.exception.code,'RECEIPT_ENVIRONMENT_CONFLICT')

if __name__=='__main__':unittest.main()
