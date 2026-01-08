## Render config (à ne pas modifier tant que tout fonctionne)

Dans Render → Web Service → Settings :

- Build Command:
  pip install -r requirements.txt && python -m py_compile main.py

- Start Command:
  uvicorn main:app --host 0.0.0.0 --port $PORT

Dans Render → Web Service → Environment :
- DATABASE_URL doit être défini (Internal Database URL de la DB Render)


## Checklist (smoke tests)

1) Installer les dépendances (local)
python -m pip install -r requirements.txt

2) Lancer l’API (local)
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

3) Health (local)
curl -sS http://127.0.0.1:8000/health

4) Health (prod)
curl -sS https://api.basketmanager-game.com/health

5) Submit (prod)
curl -i -sS -X POST "https://api.basketmanager-game.com/v1/leaderboard/season/submit" \
  -H "Content-Type: application/json" \
  -d '{"season_id":"2026-01","profile_uuid":"TEST_UUID_1234567890","pseudo":"Isidro","club":"BM Club","club_level":1,"titles_total":0,"winrate":0.0,"score_final":80,"meta":{"matches_played":0,"wins":0,"max_streak":0},"client_sig":""}'

6) Top (prod)
curl -i -sS "https://api.basketmanager-game.com/v1/leaderboard/season/top?season_id=2026-01&metric=score_final&limit=10"
