from fastapi import FastAPI

app = FastAPI(title="BM Online API", version="0.0.1")

@app.get("/")
def root():
    return {"ok": True, "service": "bm-online-api"}

@app.get("/v1/health")
def health():
    return {"ok": True}
# -------------------------------
# Cloud Save V1 (SQLite) - BM
# -------------------------------
import json
import os
import sqlite3
import time
import hashlib
import threading
from fastapi import Request, Header, HTTPException

_DB_LOCK = threading.Lock()
_DB_CONN = None

# Taille max d'un save (ajuste si besoin)
MAX_SAVE_BYTES = 512 * 1024  # 512 KB

def _db_path() -> str:
    """
    Render: si un disque est attaché, tu peux utiliser RENDER_DISK_PATH.
    Sinon on stocke dans le répertoire courant.
    """
    base = os.environ.get("RENDER_DISK_PATH", "").strip()
    if base:
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
        return os.path.join(base, "bm_cloud.sqlite")
    return "bm_cloud.sqlite"

def _get_conn():
    global _DB_CONN
    if _DB_CONN is not None:
        return _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is None:
            path = _db_path()
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cloud_saves (
                    token TEXT PRIMARY KEY,
                    save_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            conn.commit()
            _DB_CONN = conn
    return _DB_CONN

def _require_token(x_client_token: str | None) -> str:
    tok = (x_client_token or "").strip()
    if not tok:
        raise HTTPException(status_code=401, detail="Missing X-Client-Token")
    # garde simple en V1 : bornes anti-abus
    if len(tok) < 8 or len(tok) > 128:
        raise HTTPException(status_code=400, detail="Invalid X-Client-Token")
    return tok

@app.put("/v1/cloud/save")
async def cloud_save_put(
    request: Request,
    x_client_token: str | None = Header(default=None, alias="X-Client-Token"),
):
    tok = _require_token(x_client_token)

    raw = await request.body()
    if raw is None:
        raw = b""
    if len(raw) > MAX_SAVE_BYTES:
        raise HTTPException(status_code=413, detail="Save too large")

    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Normalisation (stabilise le sha)
    try:
        normalized = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        raise HTTPException(status_code=400, detail="JSON serialize error")

    data_bytes = normalized.encode("utf-8")
    if len(data_bytes) > MAX_SAVE_BYTES:
        raise HTTPException(status_code=413, detail="Save too large")

    sha = hashlib.sha256(data_bytes).hexdigest()
    now = int(time.time())

    conn = _get_conn()
    with _DB_LOCK:
        conn.execute(
            """
            INSERT INTO cloud_saves(token, save_json, sha256, size_bytes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(token) DO UPDATE SET
                save_json=excluded.save_json,
                sha256=excluded.sha256,
                size_bytes=excluded.size_bytes,
                updated_at=excluded.updated_at;
            """,
            (tok, normalized, sha, len(data_bytes), now),
        )
        conn.commit()

    return {"ok": True, "updated_at": now, "size_bytes": len(data_bytes), "sha256": sha}

@app.get("/v1/cloud/save")
def cloud_save_get(
    x_client_token: str | None = Header(default=None, alias="X-Client-Token"),
):
    tok = _require_token(x_client_token)
    conn = _get_conn()
    cur = conn.execute(
        "SELECT save_json FROM cloud_saves WHERE token=?",
        (tok,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No cloud save for this token")
    # Retour JSON direct
    try:
        return json.loads(row[0])
    except Exception:
        raise HTTPException(status_code=500, detail="Stored JSON corrupted")

@app.get("/v1/cloud/save/meta")
def cloud_save_meta(
    x_client_token: str | None = Header(default=None, alias="X-Client-Token"),
):
    tok = _require_token(x_client_token)
    conn = _get_conn()
    cur = conn.execute(
        "SELECT sha256, size_bytes, updated_at FROM cloud_saves WHERE token=?",
        (tok,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No cloud save for this token")
    return {"ok": True, "sha256": row[0], "size_bytes": int(row[1]), "updated_at": int(row[2])}

