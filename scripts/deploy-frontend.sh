#!/usr/bin/env bash
# InsightAI Frontend — Vercel Deploy Script
# Usage:
#   1. Get token: https://vercel.com/account/tokens
#   2. Run: VERCEL_TOKEN=vercel_xxx bash scripts/deploy-frontend.sh
#
# This script:
#   - Verifies Vercel auth
#   - Links the current frontend/ to your Vercel project (or creates one)
#   - Sets VITE_API_BASE_URL=https://insightai-api.fly.dev
#   - Deploys to production
#
# After it finishes, capture the production URL and:
#   fly secrets set -a insightai-api \
#     CORS_ORIGINS="https://<that-url>" \
#     CORS_ORIGINS_DEFAULT="https://<that-url>"

set -euo pipefail

if [[ -z "${VERCEL_TOKEN:-}" ]]; then
  echo "ERROR: VERCEL_TOKEN is empty."
  echo "Get a token at https://vercel.com/account/tokens then re-run with:"
  echo "  VERCEL_TOKEN=vercel_xxx bash scripts/deploy-frontend.sh"
  exit 1
fi

cd "$(dirname "$0")/.."
echo "Working in: $(pwd)"

echo
echo "=== [1/5] Verify Vercel auth ==="
vercel whoami --token "$VERCEL_TOKEN"

echo
echo "=== [2/5] Link Vercel project (creates if missing) ==="
# --yes skips interactive prompts; --team picks the team scope if you have multiple
# We assume personal scope here. Adjust if needed.
if [[ ! -d .vercel ]]; then
  vercel link --yes --token "$VERCEL_TOKEN"
else
  echo ".vercel/ already exists — skipping link (already linked)"
fi

echo
echo "=== [3/5] Set VITE_API_BASE_URL in production env ==="
echo "https://insightai-api.fly.dev" | vercel env add VITE_API_BASE_URL production --token "$VERCEL_TOKEN" --yes 2>&1 || \
  echo "  (env may already exist; will overwrite to be safe)"
vercel env rm VITE_API_BASE_URL production --token "$VERCEL_TOKEN" --yes 2>/dev/null || true
echo "https://insightai-api.fly.dev" | vercel env add VITE_API_BASE_URL production --token "$VERCEL_TOKEN" --yes

echo
echo "=== [4/5] Confirm env ==="
vercel env ls production --token "$VERCEL_TOKEN"

echo
echo "=== [5/5] Deploy to production ==="
vercel deploy --prod --yes --token "$VERCEL_TOKEN"

echo
echo "=== Done ==="
echo "Next step: copy the production URL above, then on Fly:"
echo "  fly secrets set -a insightai-api CORS_ORIGINS=\"https://<url>\" CORS_ORIGINS_DEFAULT=\"https://<url>\""
