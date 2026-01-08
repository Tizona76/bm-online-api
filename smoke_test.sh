#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-https://api.basketmanager-game.com}"
SEASON_ID="${SEASON_ID:-2026-01}"
PROFILE_UUID="${PROFILE_UUID:-TEST_UUID_1234567890}"
PSEUDO="${PSEUDO:-Isidro}"
CLUB="${CLUB:-BM Club}"
CLUB_LEVEL="${CLUB_LEVEL:-1}"
TITLES_TOTAL="${TITLES_TOTAL:-0}"
WINRATE="${WINRATE:-0.0}"
SCORE_FINAL="${SCORE_FINAL:-80}"
CLIENT_SIG="${CLIENT_SIG:-}"

echo "== Smoke test =="
echo "API_BASE:   $API_BASE"
echo "SEASON_ID:  $SEASON_ID"
echo "PROFILE:    $PROFILE_UUID"
echo

echo "[1/3] GET /health"
curl -i -sS "${API_BASE}/health"
echo
echo

echo "[2/3] POST /v1/leaderboard/season/submit"
curl -i -sS -X POST "${API_BASE}/v1/leaderboard/season/submit" \
  -H "Content-Type: application/json" \
  -d "{\"season_id\":\"${SEASON_ID}\",\"profile_uuid\":\"${PROFILE_UUID}\",\"pseudo\":\"${PSEUDO}\",\"club\":\"${CLUB}\",\"club_level\":${CLUB_LEVEL},\"titles_total\":${TITLES_TOTAL},\"winrate\":${WINRATE},\"score_final\":${SCORE_FINAL},\"meta\":{\"matches_played\":0,\"wins\":0,\"max_streak\":0},\"client_sig\":\"${CLIENT_SIG}\"}"
echo
echo

echo "[3/3] GET /v1/leaderboard/season/top"
curl -i -sS "${API_BASE}/v1/leaderboard/season/top?season_id=${SEASON_ID}&metric=score_final&limit=10"
echo
echo

echo "✅ Smoke test OK"
