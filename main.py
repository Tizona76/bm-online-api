from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import json
import time
import hashlib
import sqlite3
import threading


app = FastAPI(title="BM Online API", version="0.0.1")

@app.get("/")
def root():
    return {"ok": True, "service": "bm-online-api"}

@app.get("/v1/health")
def health():
    return {"ok": True}

# ============================================================
# LEADERBOARD (Postgres) — V1 saison mensuelle
# ============================================================
import os
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_LB_ENGINE: Optional[Engine] = None

def _lb_get_engine() -> Optional[Engine]:
    """Retourne un engine Postgres pour le leaderboard (ou None si non configuré)."""
    global _LB_ENGINE
    if _LB_ENGINE is not None:
        return _LB_ENGINE
    try:
        db_url = (os.environ.get("DATABASE_URL", "") or "").strip()
        if not db_url:
            return None
        
        # Force driver psycopg v3 (Python 3.13 compatible)
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)

        _LB_ENGINE = create_engine(db_url, pool_pre_ping=True, future=True, connect_args={"sslmode": "require"})
        return _LB_ENGINE
    except Exception as e:
        try:
            print("[DBG][LB] _lb_get_engine FAIL:", repr(e))
            du = (os.environ.get("DATABASE_URL", "") or "").strip()
            print("[DBG][LB] DATABASE_URL len =", len(du), "starts=", du[:30])
        except Exception:
            pass
        return None


def _lb_init_schema() -> bool:
    """Crée la table si besoin. Ne casse jamais le serveur Cloud."""
    eng = _lb_get_engine()
    if eng is None:
        return False
    try:
        ddl = """
        CREATE TABLE IF NOT EXISTS leaderboard_season (
          season_id       TEXT NOT NULL,
          profile_uuid    TEXT NOT NULL,
          pseudo          TEXT NOT NULL,
          club            TEXT NOT NULL,
          club_level      INT  NOT NULL DEFAULT 1,
          titles_total    INT  NOT NULL DEFAULT 0,
          winrate         DOUBLE PRECISION NOT NULL DEFAULT 0,
          score_final     INT  NOT NULL DEFAULT 0,
          meta_json       JSONB NOT NULL DEFAULT '{}'::jsonb,
          client_sig      TEXT NOT NULL DEFAULT '',
          updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          CONSTRAINT leaderboard_season_pk PRIMARY KEY (season_id, profile_uuid)
        );
        CREATE INDEX IF NOT EXISTS lb_season_score_idx
          ON leaderboard_season (season_id, score_final DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS lb_season_winrate_idx
          ON leaderboard_season (season_id, winrate DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS lb_season_titles_idx
          ON leaderboard_season (season_id, titles_total DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS lb_season_level_idx
          ON leaderboard_season (season_id, club_level DESC, updated_at DESC);
        """
        with eng.begin() as conn:
            conn.execute(text(ddl))
        return True
    except Exception:
        return False

def _lb_sig_v1(season_id: str, profile_uuid: str, pseudo: str, club: str,
              club_level: int, titles_total: int, winrate: float, score_final: int) -> str:
    """Signature simple V1 (anti-payload cassé). Si API_SALT_V1 absent => non vérifiée."""
    try:
        import hashlib
        salt = (os.environ.get("API_SALT_V1", "") or "").strip()
        canon = "|".join([
            str(season_id),
            str(profile_uuid),
            str(pseudo),
            str(club),
            str(int(club_level)),
            str(int(titles_total)),
            f"{float(winrate):.6f}",
            str(int(score_final)),
        ])
        return hashlib.sha256((canon + "|" + salt).encode("utf-8")).hexdigest()
    except Exception:
        return ""

# ============================================================
# LEADERBOARD — Endpoints Postgres (V1)
# ============================================================

class LBMeta(BaseModel):
    matches_played: int = 0
    wins: int = 0
    max_streak: int = 0
    rank_start: Optional[int] = None
    rank_end: Optional[int] = None

class LBSubmitPayload(BaseModel):
    season_id: str  # "YYYY-MM"
    profile_uuid: str
    pseudo: str
    club: str
    club_level: int = 1
    titles_total: int = 0
    winrate: float = 0.0
    score_final: int = 0
    meta: LBMeta = LBMeta()
    client_sig: Optional[str] = ""

@app.post("/v1/leaderboard/season/submit")
def lb_season_submit(p: LBSubmitPayload):
    if not _lb_init_schema():
        raise HTTPException(status_code=503, detail="LEADERBOARD_DB_NOT_READY")

    if not p.season_id or len(p.season_id) != 7 or p.season_id[4] != "-":
        raise HTTPException(status_code=400, detail="BAD_SEASON_ID")
    if not p.profile_uuid or len(p.profile_uuid) < 16:
        raise HTTPException(status_code=400, detail="BAD_PROFILE_UUID")

    salt = (os.environ.get("API_SALT_V1", "") or "").strip()
    if salt:
        expected = _lb_sig_v1(
            p.season_id, p.profile_uuid, p.pseudo, p.club,
            int(p.club_level), int(p.titles_total), float(p.winrate), int(p.score_final)
        )
        if (p.client_sig or "") != expected:
            raise HTTPException(status_code=400, detail="BAD_SIGNATURE")

    eng = _lb_get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="LEADERBOARD_DB_NOT_READY")

    import json
    meta_json = json.dumps(p.meta.model_dump())

    q = text("""
        INSERT INTO leaderboard_season
        (season_id, profile_uuid, pseudo, club, club_level, titles_total, winrate, score_final, meta_json, client_sig, updated_at)
        VALUES
        (:season_id, :profile_uuid, :pseudo, :club, :club_level, :titles_total, :winrate, :score_final, :meta_json::jsonb, :client_sig, NOW())
        ON CONFLICT (season_id, profile_uuid)
        DO UPDATE SET
            pseudo = EXCLUDED.pseudo,
            club = EXCLUDED.club,
            club_level = EXCLUDED.club_level,
            titles_total = EXCLUDED.titles_total,
            winrate = EXCLUDED.winrate,
            score_final = EXCLUDED.score_final,
            meta_json = EXCLUDED.meta_json,
            client_sig = EXCLUDED.client_sig,
            updated_at = NOW()
        RETURNING updated_at;
    """)

    with eng.begin() as conn:
        row = conn.execute(q, {
            "season_id": p.season_id,
            "profile_uuid": p.profile_uuid,
            "pseudo": (p.pseudo or "")[:24],
            "club": (p.club or "")[:24],
            "club_level": int(p.club_level),
            "titles_total": int(p.titles_total),
            "winrate": float(p.winrate),
            "score_final": int(p.score_final),
            "meta_json": meta_json,
            "client_sig": (p.client_sig or "")[:128],
        }).fetchone()

    return {"ok": True, "season_id": p.season_id, "profile_uuid": p.profile_uuid, "updated_at": str(row[0]) if row else None}

@app.get("/v1/leaderboard/season/top")
def lb_season_top(season_id: str, metric: str = "score_final", limit: int = 50, offset: int = 0):
    if not _lb_init_schema():
        raise HTTPException(status_code=503, detail="LEADERBOARD_DB_NOT_READY")

    metric = (metric or "").strip().lower()
    if metric not in ("score_final", "winrate", "titles_total", "club_level"):
        raise HTTPException(status_code=400, detail="BAD_METRIC")

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    eng = _lb_get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="LEADERBOARD_DB_NOT_READY")

    q = text(f"""
        SELECT profile_uuid, pseudo, club, club_level, titles_total, winrate, score_final, updated_at
        FROM leaderboard_season
        WHERE season_id = :season_id
        ORDER BY {metric} DESC, updated_at DESC
        LIMIT :limit OFFSET :offset;
    """)
    qc = text("SELECT COUNT(*) FROM leaderboard_season WHERE season_id = :season_id;")

    with eng.begin() as conn:
        total = int(conn.execute(qc, {"season_id": season_id}).scalar() or 0)
        rows = conn.execute(q, {"season_id": season_id, "limit": limit, "offset": offset}).fetchall()

    items = []
    for i, r in enumerate(rows):
        items.append({
            "rank": offset + i + 1,
            "profile_uuid": r[0],
            "pseudo": r[1],
            "club": r[2],
            "club_level": int(r[3]),
            "titles_total": int(r[4]),
            "winrate": float(r[5]),
            "score_final": int(r[6]),
            "updated_at": str(r[7]),
        })

    return {"ok": True, "season_id": season_id, "metric": metric, "items": items, "total": total}

@app.get("/v1/leaderboard/season/me")
def lb_season_me(season_id: str, profile_uuid: str, metric: str = "score_final"):
    if not _lb_init_schema():
        raise HTTPException(status_code=503, detail="LEADERBOARD_DB_NOT_READY")

    metric = (metric or "").strip().lower()
    if metric not in ("score_final", "winrate", "titles_total", "club_level"):
        raise HTTPException(status_code=400, detail="BAD_METRIC")

    eng = _lb_get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="LEADERBOARD_DB_NOT_READY")

    q = text(f"""
        WITH ranked AS (
            SELECT
              profile_uuid, pseudo, club, club_level, titles_total, winrate, score_final, updated_at,
              ROW_NUMBER() OVER (ORDER BY {metric} DESC, updated_at DESC) AS rk
            FROM leaderboard_season
            WHERE season_id = :season_id
        )
        SELECT profile_uuid, pseudo, club, club_level, titles_total, winrate, score_final, updated_at, rk
        FROM ranked
        WHERE profile_uuid = :profile_uuid
        LIMIT 1;
    """)

    with eng.begin() as conn:
        r = conn.execute(q, {"season_id": season_id, "profile_uuid": profile_uuid}).fetchone()

    if not r:
        return {"ok": True, "season_id": season_id, "metric": metric, "me": None}

    me = {
        "rank": int(r[8]),
        "profile_uuid": r[0],
        "pseudo": r[1],
        "club": r[2],
        "club_level": int(r[3]),
        "titles_total": int(r[4]),
        "winrate": float(r[5]),
        "score_final": int(r[6]),
        "updated_at": str(r[7]),
    }
    return {"ok": True, "season_id": season_id, "metric": metric, "me": me}

