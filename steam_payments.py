"""A4 server-only MicroTxn, disabled by default; no startup I/O or DDL.
Official contract: https://partner.steamgames.com/doc/webapi/ISteamMicroTxn
Init v3 / Finalize v2 / Query v3 / Report v5. EUR only, one pack per order.
A durable attempt flag precedes each mutating Valve call. After an ambiguous
Finalize we ONLY query (manual review if still Approved), never recapture.
GetReport reconciliation is an operator-invoked method, not a public HTTP route
or a startup job. Cursor windows overlap; every event is rechecked with Query.
"""
import json
import math
import os
import re
import secrets
import time
import urllib.parse
import urllib.request
import ssl
import certifi
from contextlib import contextmanager
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.concurrency import run_in_threadpool

if __package__:
    from . import steam_wallet as sw
    from .steam_identity import APP_ID, NoRedirect, Rejected
else:
    import steam_wallet as sw
    from steam_identity import APP_ID, NoRedirect, Rejected

CATALOG = {'tokens25': (25, 99, 1), 'tokens75': (75, 299, 2), 'tokens150': (150, 499, 3)}
REVERSED = {'Refunded', 'Chargedback', 'RefundedSuspectedFraud', 'RefundedFriendlyFraud'}
VERSIONS = {'InitTxn': ('POST', 3), 'FinalizeTxn': ('POST', 2), 'QueryTxn': ('GET', 3),
            'GetReport': ('GET', 5), 'GetUserInfo': ('GET', 2)}


def environment():
    value = os.environ.get('STEAM_PAYMENTS_ENVIRONMENT', 'sandbox')
    if value not in ('sandbox', 'live'):
        sw.reject(503, 'PAYMENTS_CONFIGURATION')
    if value == 'live' and os.environ.get('STEAM_PAYMENTS_LIVE_ENABLED') != '1':
        sw.reject(503, 'LIVE_PAYMENTS_DISABLED')
    return value


def uint64(value):
    return isinstance(value, str) and re.fullmatch('[1-9][0-9]{0,19}', value) and int(value) < 2**64


class Valve:
    def __init__(self, key_provider=None, opener=None):
        self.key_provider = key_provider or (lambda: os.environ.get('STEAM_PUBLISHER_KEY', ''))
        self.opener = opener  # even HTTP handler construction deferred until call

    def call(self, method, env, params):
        try:
            if method not in VERSIONS or env not in ('sandbox', 'live') or params.get('appid') != APP_ID:
                raise ValueError()
            key = self.key_provider()
            if not isinstance(key, str) or not key:
                raise ValueError()
            verb, version = VERSIONS[method]
            interface = 'ISteamMicroTxnSandbox' if env == 'sandbox' else 'ISteamMicroTxn'
            url = f'https://partner.steam-api.com/{interface}/{method}/v{version}/'
            payload = urllib.parse.urlencode(dict(params, key=key)).encode()
            req = urllib.request.Request(url + ('?' + payload.decode() if verb == 'GET' else ''),
                    data=payload if verb == 'POST' else None, method=verb,
                    headers={'Accept':'application/json','Content-Type':'application/x-www-form-urlencoded'})
            with (self.opener or urllib.request.build_opener(
                NoRedirect(),
                urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())),
            )).open(req, timeout=5) as reply:
                raw = reply.read(2_000_001)
                if reply.status != 200 or len(raw) > 2_000_000:
                    raise ValueError()
                return json.loads(raw)
        except Exception:
            raise Rejected(503, 'VALVE_UNAVAILABLE') from None


class ReceiptSource:
    """One registry, read on A5's existing transaction after its checkpoint lock."""
    def lock(self, conn, receipt_id):
        row = conn.execute(text('''SELECT r.receipt_id,o.user_id,o.profile_uuid,o.career_id,
            o.steam_id,o.app_id,o.environment,o.quantity,r.state FROM steam_receipts r
            JOIN steam_orders o ON o.order_id=r.order_id WHERE r.receipt_id=:r FOR UPDATE OF r'''),
            {'r':receipt_id}).mappings().first()
        return dict(row) if row else None


class Payments:
    def __init__(self, engine_provider, valve=None, env='sandbox', *, steam_id=None):
        if env not in ('sandbox', 'live'): sw.reject(503, 'PAYMENTS_CONFIGURATION')
        self.wallet = sw.Wallet(engine_provider, steam_id=steam_id)
        self.valve = valve or Valve()
        self.env = env

    def rpc(self, method, params):
        try:
            value = self.valve.call(method, self.env, dict(params, appid=APP_ID))
            response = value.get('response') if isinstance(value, dict) else None
            if not isinstance(response, dict) or response.get('result') != 'OK' or not isinstance(response.get('params'), dict):
                sw.reject(503, 'VALVE_UNCONFIRMED')
            return response['params']
        except Exception:
            raise Rejected(503, 'VALVE_UNCONFIRMED') from None

    def order(self, conn, user, oid, lock=False):
        if not uint64(oid) or int(oid) >= 2**63: sw.reject(404, 'ORDER_NOT_FOUND')
        row = conn.execute(text('SELECT * FROM steam_orders WHERE order_id=:id AND user_id=:u AND environment=:e'
                         + (' FOR UPDATE' if lock else '')), {'id':int(oid),'u':user,'e':self.env}).mappings().first()
        if not row: sw.reject(404, 'ORDER_NOT_FOUND')
        return dict(row)

    def public(self, conn, row):
        receipt = conn.execute(text('SELECT receipt_id,state FROM steam_receipts WHERE order_id=:id'), {'id':row['order_id']}).mappings().first()
        return {k:row[k] for k in ('state','pack_id','profile_uuid','career_id','currency','amount_minor')} | {
            'order_id':str(row['order_id']), 'receipt':dict(receipt) if receipt else None}

    def get(self, user, oid):
        with self.wallet.transaction() as conn:
            self.wallet.authenticate(conn, user)
            return 200, self.public(conn, self.order(conn, user, oid))

    def create(self, user, body, key):
        sw.fields(body, ('pack_id','profile_uuid','career_id','language'))
        sw.identifier(key); sw.identifier(body['profile_uuid']); sw.identifier(body['career_id'])
        if body['pack_id'] not in CATALOG or body['language'] not in ('en','fr','es','de','it','pt'):
            sw.reject(400, 'INVALID_CATALOG_SELECTION')
        request_hash = sw.fingerprint(body)
        with self.wallet.transaction() as conn:
            steam = self.wallet.authenticate(conn,user)
            old = conn.execute(text('SELECT * FROM steam_orders WHERE user_id=:u AND idempotency_key=:k'), {'u':user,'k':key}).mappings().first()
            if old:
                if old['request_hash'] != request_hash or old['environment'] != self.env: sw.reject(409, 'IDEMPOTENCY_CONFLICT')
                return 200, self.public(conn,old)
            # Require A5 enrollment before accepting money, not just an unverified client destination.
            args = {'u':user,'p':body['profile_uuid'],'c':body['career_id']}
            cp = self.wallet.checkpoint(conn,args)
            if not cp: sw.reject(409,'CHECKPOINT_REQUIRED')
            if conn.execute(text("SELECT 1 FROM steam_orders WHERE user_id=:u AND state IN ('NEW','INIT_UNKNOWN','AWAITING','FINALIZE_UNKNOWN')"),args).first():
                sw.reject(409,'ORDER_PENDING')
            qty, amount, item = CATALOG[body['pack_id']]
            oid = secrets.randbelow(2**63-1)+1
            values = args | {'id':oid,'s':steam,'e':self.env,'k':key,'h':request_hash,'q':qty,'a':amount,'i':item,'pack':body['pack_id'],'lang':body['language']}
            conn.execute(text('''INSERT INTO steam_orders(order_id,user_id,profile_uuid,career_id,steam_id,
                environment,idempotency_key,request_hash,quantity,amount_minor,item_id,pack_id,language)
                VALUES(:id,:u,:p,:c,:s,:e,:k,:h,:q,:a,:i,:pack,:lang)'''),values)
        self.initialize(user,str(oid))
        status, result = self.get(user,str(oid))
        return (202 if result['state']=='INIT_UNKNOWN' else 201), result

    @contextmanager
    def exclusive(self, oid):
        # Session advisory lock spans bounded network I/O, not a SQL transaction.
        # Attempt flags survive process death / loss of this connection.
        engine = self.wallet.identity.engine()
        if engine.dialect.name != 'postgresql': sw.reject(503,'POSTGRES_REQUIRED')
        with engine.connect() as lock:
            acquired = lock.execute(text('SELECT pg_try_advisory_lock(:id)'), {'id':int(oid)}).scalar()
            lock.commit()
            if not acquired: sw.reject(409,'ORDER_BUSY')
            try: yield
            finally:
                try:
                    lock.execute(text('SELECT pg_advisory_unlock(:id)'), {'id':int(oid)}); lock.commit()
                except Exception:
                    lock.invalidate()

    def mark(self,user,oid,state,expected=None,**values):
        with self.wallet.transaction() as conn:
            self.wallet.authenticate(conn,user)
            current=self.order(conn,user,oid,True)
            if expected is not None and current['state'] not in expected: return False
            conn.execute(text('UPDATE steam_orders SET state=:state'+''.join(','+k+'=:'+k for k in values)+' WHERE order_id=:id'),
                         dict(values,state=state,id=int(oid)))
            return True

    def initialize(self,user,oid):
        with self.exclusive(oid):
            with self.wallet.transaction() as conn:
                self.wallet.authenticate(conn,user); row=self.order(conn,user,oid)
            if row['state']!='NEW': return
            info=self.rpc('GetUserInfo',{'steamid':row['steam_id']})
            # Valve converts the EUR order price to the user's wallet currency.
            if info.get('status') not in ('Active','Trusted'):
                self.mark(user,oid,'FAILED',expected=('NEW',)); sw.reject(409,'STEAM_PURCHASE_UNAVAILABLE')
            if not self.mark(user,oid,'INIT_UNKNOWN',expected=('NEW',)): return  # atomic claim BEFORE Valve
            try:
                reply=self.rpc('InitTxn',{'orderid':oid,'steamid':row['steam_id'],'itemcount':1,
                    'language':row['language'],'currency':'EUR','usersession':'client','itemid[0]':row['item_id'],
                    'qty[0]':1,'amount[0]':row['amount_minor'],'description[0]':f"{row['quantity']} Club Tokens"})
                if reply.get('orderid')!=oid or not uint64(reply.get('transid')): return
                self.mark(user,oid,'AWAITING',expected=('INIT_UNKNOWN',),trans_id=reply['transid'])
            except Rejected:
                return  # unknown stays durable, no second InitTxn

    def verify(self, params, row):
        if params.get('orderid')!=str(row['order_id']) or params.get('steamid')!=row['steam_id'] or params.get('currency')!='EUR' or not uint64(params.get('transid')):
            sw.reject(502,'VALVE_CONTEXT_MISMATCH')
        if row['trans_id'] and row['trans_id']!=params['transid']: sw.reject(502,'VALVE_CONTEXT_MISMATCH')
        if 'appid' in params and params['appid']!=APP_ID: sw.reject(502,'VALVE_CONTEXT_MISMATCH')
        items=params.get('items')
        if not isinstance(items,list) or len(items)!=1 or not isinstance(items[0],dict): sw.reject(502,'VALVE_ITEMS_MISMATCH')
        item=items[0]
        if type(item.get('itemid')) is not int or item['itemid']!=row['item_id'] or type(item.get('qty')) is not int or item['qty']!=1:
            sw.reject(502,'VALVE_ITEMS_MISMATCH')
        if any(type(item.get(k)) is not int or item[k]<0 for k in ('amount','vat')) or item['amount']+item['vat']!=row['amount_minor']:
            sw.reject(502,'VALVE_TOTAL_MISMATCH')
        return params

    def reconcile(self,user,oid,authorize=False,cancel=False):
        # A callback is only a hint. A GET route never calls this method.
        with self.wallet.transaction() as conn:
            self.wallet.authenticate(conn,user); self.order(conn,user,oid)
        with self.exclusive(oid):
            with self.wallet.transaction() as conn: row=self.order(conn,user,oid)
            if row['state']=='NEW':
                # No Init has ever been attempted. Explicit resume may initialize later.
                return 202,self.public_read(user,oid)
            if cancel and row['state'] in ('INIT_UNKNOWN','AWAITING'):
                if not self.mark(user,oid,'CANCELLED',expected=('INIT_UNKNOWN','AWAITING')): return self.get(user,oid)
                row['state']='CANCELLED'
            try:
                params=self.verify(self.rpc('QueryTxn',{'orderid':oid}),row)
                if params['status']=='Approved' and authorize and row['state'] in ('INIT_UNKNOWN','AWAITING'):
                    if not self.mark(user,oid,'FINALIZE_UNKNOWN',expected=('INIT_UNKNOWN','AWAITING'),trans_id=params['transid']):
                        return self.get(user,oid)
                    row['state']='FINALIZE_UNKNOWN';row['trans_id']=params['transid']
                    try: self.rpc('FinalizeTxn',{'orderid':oid})
                    except Rejected: pass
                    params=self.verify(self.rpc('QueryTxn',{'orderid':oid}),row)
                self.apply(user,oid,params)
            except Rejected as exc:
                if exc.status==502: raise
                return 202,self.public_read(user,oid)
        return self.get(user,oid)

    def public_read(self,user,oid):
        return self.get(user,oid)[1]

    def apply(self,user,oid,params):
        # Shared lock order: user -> checkpoint -> order -> receipt -> A5 delivery.
        with self.wallet.transaction() as conn:
            self.wallet.authenticate(conn,user)
            row=self.order(conn,user,oid)
            self.wallet.checkpoint(conn,{'u':user,'p':row['profile_uuid'],'c':row['career_id']})
            row=self.order(conn,user,oid,True);self.verify(params,row)
            receipt=conn.execute(text('SELECT * FROM steam_receipts WHERE order_id=:id FOR UPDATE'),{'id':int(oid)}).mappings().first()
            status=params.get('status'); itemstatus=params['items'][0].get('itemstatus')
            if status in REVERSED or status=='PartialRefund':
                state='REVIEW' if status=='PartialRefund' else 'REVOKED'
                if receipt:
                    reserved=conn.execute(text('SELECT 1 FROM steam_wallet_deliveries WHERE receipt_id=:r'),{'r':receipt['receipt_id']}).first()
                    rs='manual' if reserved or status=='PartialRefund' else 'revoked'
                    conn.execute(text('UPDATE steam_receipts SET state=:s WHERE order_id=:id'),{'s':rs,'id':int(oid)})
            elif row['state'] in ('REVOKED','REVIEW'):
                return  # never resurrect reversed entitlements
            elif status=='Succeeded' and itemstatus=='Succeeded':
                state='PAID'
                conn.execute(text("UPDATE steam_orders SET state='PAID',trans_id=:t WHERE order_id=:id"),{'t':params['transid'],'id':int(oid)})
                conn.execute(text('''INSERT INTO steam_receipts(receipt_id,order_id,state)
                    VALUES(:r,:id,'confirmed') ON CONFLICT(order_id) DO NOTHING'''),{'r':'steam-'+self.env+'-'+oid,'id':int(oid)})
            elif receipt:
                return  # old Init/Approved responses never erase a confirmed receipt
            elif status=='Failed': state='FAILED'
            elif row['state']=='CANCELLED': state='CANCELLED'
            elif status in ('Approved','Init'): state='FINALIZE_UNKNOWN' if row['state']=='FINALIZE_UNKNOWN' else 'AWAITING'
            else: state='REVIEW'
            conn.execute(text('UPDATE steam_orders SET state=:s,trans_id=:t WHERE order_id=:id'),{'s':state,'t':params['transid'],'id':int(oid)})

    def receipts(self,user,body):
        sw.fields(body,('profile_uuid','career_id','cursor'))
        sw.identifier(body['profile_uuid']);sw.identifier(body['career_id'])
        cursor=body['cursor']
        if not isinstance(cursor,str) or not re.fullmatch('[0-9]{1,19}',cursor) or int(cursor)>=2**63: sw.reject(400,'INVALID_CURSOR')
        with self.wallet.transaction() as conn:
            self.wallet.authenticate(conn,user)
            rows=conn.execute(text('''SELECT r.receipt_id,r.state,r.receipt_seq,o.order_id FROM steam_receipts r JOIN steam_orders o USING(order_id)
                WHERE o.user_id=:u AND o.profile_uuid=:p AND o.career_id=:c AND o.environment=:e AND r.receipt_seq>:cursor
                ORDER BY r.receipt_seq LIMIT 100'''),{'u':user,'p':body['profile_uuid'],'c':body['career_id'],'e':self.env,'cursor':int(cursor)}).mappings().all()
            return 200,{'receipts':[{'receipt_id':r['receipt_id'],'state':r['state'],'order_id':str(r['order_id'])} for r in rows],
                        'next_cursor':str(rows[-1]['receipt_seq']) if rows else cursor}

    def reconcile_report(self,start):
        """Operator only. Replay overlapping windows; never trust a client report.
        No scheduler, HTTP admin route, or automatic debit. Use the returned latest event time for an overlapping next window.
        Short nonempty batches do not imply completeness; equal-time saturation
        requires operator review / a larger bounded batch, never skipping ahead.
        """
        datetime.strptime(start,'%Y-%m-%dT%H:%M:%SZ')
        report=self.rpc('GetReport',{'time':start,'type':'CHARGEBACK','maxresults':1000})
        rows=report.get('orders')
        if not isinstance(rows,list) or report.get('count')!=len(rows): sw.reject(502,'REPORT_INVALID')
        times=[]
        for event in rows:
            try:
                stamp=event['time']
                datetime.strptime(stamp,'%Y-%m-%dT%H:%M:%SZ')
                if stamp < start: raise ValueError()
                times.append(stamp)
            except (KeyError,TypeError,ValueError): sw.reject(502,'REPORT_INVALID')
            if not isinstance(event,dict) or not uint64(event.get('orderid')): sw.reject(502,'REPORT_INVALID')
            with self.wallet.transaction() as conn:
                found=conn.execute(text('SELECT user_id,steam_id FROM steam_orders WHERE order_id=:id AND environment=:e'),{'id':int(event['orderid']),'e':self.env}).first()
            if found:
                # Revalidate the stored Steam identity through the existing A3/A5 checks.
                service=Payments(self.wallet.identity.engine_provider,self.valve,self.env,steam_id=found[1])
                code,_=service.reconcile(found[0],event['orderid'])
                if code!=200: sw.reject(503,'REPORT_PENDING')
        with self.wallet.transaction() as conn:
            conn.execute(text('''INSERT INTO steam_reconciliation(environment,window_start,event_count,complete,next_window)
                VALUES(:e,:s,:n,:done,:next) ON CONFLICT(environment,window_start) DO UPDATE
                SET event_count=EXCLUDED.event_count,complete=EXCLUDED.complete,next_window=EXCLUDED.next_window'''),
                {'e':self.env,'s':start,'n':len(rows),'done':not rows,'next':max(times) if times else start})
        return {'processed':len(rows),'complete':not rows,'next_window':max(times) if times else start}


def create_router(engine_provider,require_bearer,*,valve=None,enabled=None,purchase_sessions=False):
    router=APIRouter()
    active=enabled or (lambda: os.environ.get('STEAM_PAYMENTS_ENABLED')=='1')
    async def call(request,action,oid=None):
        try:
            if not active(): sw.reject(503,'STEAM_PAYMENTS_DISABLED')
            claims=require_bearer(request.headers.get('authorization',''))
            if not isinstance(claims,dict) or not isinstance(claims.get('sub'),str) or not claims['sub']: sw.reject(401,'INVALID_SESSION')
            exp=claims.get('exp')
            if type(exp) not in (float,int) or not math.isfinite(exp) or exp<=time.time(): sw.reject(401,'INVALID_SESSION')
            body={}
            if request.method=='GET': body=dict(request.query_params)
            else:
                if request.headers.get('content-type','').split(';')[0]!='application/json': sw.reject(415,'JSON_REQUIRED')
                raw=bytearray()
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw)>4096: sw.reject(413,'REQUEST_TOO_LARGE')
                def unique(pairs):
                    out={}
                    for k,v in pairs:
                        if k in out: sw.reject(400,'INVALID_JSON')
                        out[k]=v
                    return out
                body=json.loads(raw,object_pairs_hook=unique)
            steam_id=claims.get('steam_id') if purchase_sessions and claims.get('typ')=='steam_purchase' else None
            service=Payments(engine_provider,valve,environment(),steam_id=steam_id);user=claims['sub']
            def execute():
                if action=='catalog':
                    with service.wallet.transaction() as conn: service.wallet.authenticate(conn,user)
                    return 200,{'currency':'EUR','version':1,'packs':[{'pack_id':p,'tokens':v[0],'amount_minor':v[1]} for p,v in CATALOG.items()]}
                if action=='create': return service.create(user,body,request.headers.get('idempotency-key'))
                if action=='get': return service.get(user,oid)
                if action=='receipts': return service.receipts(user,body)
                if action=='authorization':
                    sw.fields(body,('authorized',))
                    if type(body['authorized']) is not bool: sw.reject(400,'INVALID_REQUEST')
                    return service.reconcile(user,oid,body['authorized'],not body['authorized'])
                sw.fields(body,())
                # Resume is an explicit user confirmation action, never a GET.
                with service.wallet.transaction() as conn:
                    service.wallet.authenticate(conn,user);row=service.order(conn,user,oid)
                if row['state']=='NEW': service.initialize(user,oid)
                return service.reconcile(user,oid,True)
            status,payload=await run_in_threadpool(execute)
            return JSONResponse(payload,status_code=status,headers={'Cache-Control':'no-store'})
        except Rejected as exc: status,code=exc.status,exc.code
        except HTTPException: status,code=401,'INVALID_SESSION'
        except IntegrityError: status,code=409,'PAYMENT_CONFLICT'
        except SQLAlchemyError: status,code=503,'PAYMENT_STORE_UNAVAILABLE'
        except (ValueError,TypeError,KeyError,UnicodeError,RecursionError): status,code=400,'INVALID_REQUEST'
        except Exception: status,code=503,'PAYMENTS_UNAVAILABLE'
        return JSONResponse({'detail':code},status_code=status,headers={'Cache-Control':'no-store'})
    @router.get('/v1/steam/payments/catalog')
    async def catalog(request:Request): return await call(request,'catalog')
    @router.post('/v1/steam/payments/orders')
    async def create(request:Request): return await call(request,'create')
    @router.get('/v1/steam/payments/orders/{order_id}')
    async def get(request:Request,order_id:str): return await call(request,'get',order_id)
    @router.post('/v1/steam/payments/orders/{order_id}/authorization')
    async def authorization(request:Request,order_id:str): return await call(request,'authorization',order_id)
    @router.post('/v1/steam/payments/orders/{order_id}/resume')
    async def resume(request:Request,order_id:str): return await call(request,'resume',order_id)
    @router.get('/v1/steam/payments/receipts')
    async def receipts(request:Request): return await call(request,'receipts')
    return router
