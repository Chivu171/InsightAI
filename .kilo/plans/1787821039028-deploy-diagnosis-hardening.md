# InsightAI — Deployment Diagnosis & Hardening Plan

## Context

User has provisioned Fly.io account + app name, Vercel project link, OpenRouter key, Upstash Redis, and GitHub Actions secrets. The app has been deployed partially (status unclear) and is failing. User asked for a self-driven diagnosis. This plan enumerates the **highest-likelihood failure modes** based on config review (no live log access from plan agent), gives a reproducible diagnostic procedure, and lists the targeted fixes a Builder should apply.

Target stack: **Fly.io (backend, `insightai-api`) + Vercel (frontend) + Upstash Redis + OpenRouter**. Persistent volume `/data` on Fly for vector_db + HF cache.

## Constraints

- Do **not** invent new infra (no new providers).
- Do **not** change tech stack — only harden what's already there.
- All fixes must be **reversible** (env-only or feature-flagged where possible).
- Preserve existing `DEPLOY.md` instructions; this plan **complements** it.
- Keep CI workflow (`deploy.yml`) the single source of truth for prod deploys.

---

## Likely Failure Modes (ranked by probability)

| # | Failure mode | Evidence in repo | Symptom |
|---|---|---|---|
| F1 | **CORS hard-fail in prod** | `backend/main.py:19-22` raises `RuntimeError("CORS_ORIGINS is required in production")` if env empty. `fly.toml` only **comments** that it must be set — no validation that it is. | 500 on every request from browser, or app crashes at boot. |
| F2 | **First-boot healthcheck timeout** | `fly.toml` `grace_period = "60s"`; `Dockerfile` `start-period=60s`; HF download of `bge-small-en-v1.5` (~120MB) + `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) + `pip install` of langchain stack on every deploy. | Fly kills the machine mid-boot, never reaches `/health` 200, deploy appears to "succeed" but app is unreachable. |
| F3 | **Volume not created** | `fly.toml` declares `[[mounts]] source = "insightai_data"`, but `fly launch --no-deploy --copy-config` (per `DEPLOY.md`) does **not** create the volume. First deploy mounts an empty dir, then on every restart vector_db is wiped. | Index disappears after each redeploy; users see "no data" errors. |
| F4 | **OpenRouter free-model queue** | `config.py:45` default `deepseek/deepseek-v4-flash:free`. Free models on OpenRouter are queued; from Vietnam latency often > 60s. `query_router._classify_query` and `rewrite_query_with_history` add 1–2 extra LLM calls **before** the answer stream even starts. | Chat appears hung; first token takes minutes. |
| F5 | **Race between `/reset` and background `/upload` task** | `documents.py:91-95` does `clear_index()` + `shutil.rmtree(vector_db_path)` synchronously while the `/upload` `BackgroundTasks` process is still writing. No file lock. | Intermittent 500 / corrupted index / "Could not load index" on next boot. |
| F6 | **`conversation_store.clear()` nukes all Upstash sessions** | `indexing.clear_index(engine)` (called on every upload) → `engine.conversation_store.clear()` → Upstash `SCAN MATCH "session:*"` + `DEL` of every key. | Other users' chat histories vanish when anyone uploads a file. Multi-tenant data loss. |
| F7 | **Frontend built with stale `VITE_API_BASE_URL`** | Vite bakes env at build time. If `VITE_API_BASE_URL` is added/changed in Vercel after first deploy, the new value is ignored until a forced redeploy. | Frontend still calls `http://localhost:8000` in production → CORS error or net::ERR. |
| F8 | **Debug endpoints exposed in prod** | `chat.py:20-67` exposes `/stream-test` and `/stream-test-sync` unconditionally. | Minor — info leak / DoS surface. |
| F9 | **`api.ts` timeout constant bug** | `LONG_RUNNING_TIMEOUT_MS_LOCAL = 20 * 60 * 100000` = 1.2 × 10⁹ ms ≈ 14 days. Probably a copy-paste error (should be `20 * 60 * 1000`). | Local dev hangs forever on hung requests; harmless in prod (only applies to localhost). |
| F10 | **`google-genai` / `streamlit` dead deps in requirements.txt** | `requirements.txt:7,18` lists them, but `generate_with_google` is **not called** in `generate_text` (only OpenRouter + LMStudio). | Larger Docker image, slower pip install, larger attack surface. |

---

## Diagnostic Procedure (run first, before any fix)

A Builder (or user with `fly`/`vercel` CLIs) should execute these in order and report findings. Each step is independent.

```
# 1. Confirm Fly app exists and is running
fly status -a insightai-api

# 2. Confirm volume exists and is attached
fly volumes list -a insightai-api
# Expect: 1 volume, name=insightai_data, region=sin, attached=insightai-api

# 3. Confirm secrets are set
fly secrets list -a insightai-api
# Expect: OPENROUTER_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, CORS_ORIGINS

# 4. Live healthcheck (no model load)
curl -s https://insightai-api.fly.dev/health
curl -s https://insightai-api.fly.dev/status

# 5. Recent logs (most useful single command)
fly logs -a insightai-api --no-tail
# Look for: tracebacks, "CORS_ORIGINS is required", "ModuleNotFoundError", "OutOfMemoryError", HF download progress

# 6. Confirm CORS actually allows the Vercel origin
curl -i -H "Origin: https://<your-app>.vercel.app" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS https://insightai-api.fly.dev/queryHybrid

# 7. Confirm HF cache is warm on the volume
fly ssh console -a insightai-api -C "ls -la /data/hf_cache" -C "du -sh /data/hf_cache"

# 8. Vercel env check
vercel env ls --prod
# Expect: VITE_API_BASE_URL = https://insightai-api.fly.dev

# 9. Trigger rebuild on Vercel after env change (only if step 8 needed a fix)
vercel deploy --prod --force
```

**If `/health` returns 200 but `/status` shows `no_index`** → F3 or F2 (cold start, no documents uploaded yet → expected).
**If `/health` never returns 200** → F2 (timeout) or volume not mounted.
**If logs show `CORS_ORIGINS is required in production`** → F1.
**If frontend shows `Network Error` to `http://localhost:8000`** → F7.
**If chat hangs > 30s before first token** → F4.

---

## Fix Plan (ordered by impact / safety)

### Step 1 — Verify and (re-)set secrets on Fly (covers F1, partially F4)
```bash
fly secrets set -a insightai-api \
  CORS_ORIGINS="https://<your-vercel-app>.vercel.app" \
  OPENROUTER_API_KEY=<key> \
  UPSTASH_REDIS_REST_URL=<url> \
  UPSTASH_REDIS_REST_TOKEN=<token>
```
Do **not** set `GOOGLE_API_KEY` unless actually using Gemini fallback — leaving it unset makes the missing-provider path explicit.

### Step 2 — Create / verify Fly volume (covers F3)
```bash
fly volumes create insightai_data -a insightai-api --region sin --size 2 --yes
fly volumes list -a insightai-api   # confirm attached
```
If the volume exists but the index still disappears on redeploy, suspect that `fly.toml` `VECTOR_DB_PATH` env is not being respected — verify with `fly ssh console -a insightai-api -C "printenv | grep -E 'VECTOR_DB|UPLOAD|HF_HOME'"`.

### Step 3 — Bump `grace_period` and add boot log marker (covers F2)
Edit `backend/fly.toml`:
- `grace_period = "180s"` (3 minutes; first boot downloads HF models)
- `auto_stop_machines = "suspend"` is **not** used here (already `off`); keep that.

Edit `backend/Dockerfile`:
- `HEALTHCHECK --start-period=180s` (was 60s).
- Add a `RUN echo "build sha=$(date +%s)" > /app/BUILD_SHA` so a fresh deploy is identifiable in logs.

### Step 4 — Stop wiping Upstash on every upload (covers F6)
Edit `backend/rag/indexing.py` `clear_index()`: do **not** call `engine.conversation_store.clear()`. Sessions belong to users, not to the index.

Either:
- Remove the `self.conversation_store.clear()` line from `clear_index()` entirely, **or**
- Add an explicit `engine.conversation_store.clear_session(session_id)` and call it only from a per-session endpoint (not implemented in this plan — out of scope).

### Step 5 — Make `/reset` and background `/upload` non-racy (covers F5)
Edit `backend/api/routers/documents.py`:
- Add a module-level `asyncio.Lock` (or `threading.Lock` since background task is sync) around the `process()` body of `/upload`.
- `POST /reset` should `await lock` before `rmtree`; or refuse if an upload is in progress and return `409 Conflict`.
- Simplest fix: gate `/reset` and `/upload` behind the same lock; return `409` from `/reset` if `engine.status == "processing"`.

### Step 6 — Surface CORS failure as 4xx, not boot crash (covers F1)
Edit `backend/main.py`: instead of `raise RuntimeError(...)`, log a clear `ERROR` and return `cors_origins = []` is **wrong** (empty list = no CORS). Correct path: if prod and `CORS_ORIGINS` empty, set it to the configured fallback **list** from `settings.cors_origins_default` (a new field in `config.py`) which the deployer must populate. Or — simpler — keep the raise but make the error message actionable: include the `fly secrets set` command in the error.

### Step 7 — Disable debug endpoints in prod (covers F8)
Edit `backend/api/routers/chat.py`: gate `/stream-test*` behind `if settings.env != "production":` and return `404` otherwise.

### Step 8 — Fix frontend timeout constant (covers F9)
Edit `frontend/src/app/api.ts` line 27:
```ts
const LONG_RUNNING_TIMEOUT_MS_LOCAL = 20 * 60 * 1000; // 20 minutes
```
This is a dev-only constant; harmless in prod, but causes confusing local-dev "hangs".

### Step 9 — Trim dead dependencies (covers F10)
Edit `backend/requirements.txt`: remove `streamlit` and `google-genai` if and only if you confirm via `grep` that no production code path uses them. **If** a fallback to Gemini is wanted in future, keep `google-genai`; remove `streamlit` regardless. Run `pip check` after.

### Step 10 — Add `/ready` endpoint that gates on model load
Edit `backend/api/routers/health.py`: add a `GET /ready` that returns 200 only when `app.state.rag_engine.embeddings` is initialized. Change `fly.toml` `path = "/health"` to `path = "/ready"` so Fly won't mark the machine healthy before models are warm.

### Step 11 — Document the bootstrap order in `DEPLOY.md`
Append a "First-time bootstrap" section listing steps 1–3 in order, and a "Common errors" table mapping each fly-log line to the F-number from this plan. The user will not remember which secret was missing.

---

## Validation Plan

After each fix:

1. **Pre-deploy local check** (only for code changes, not env):
   ```bash
   cd backend && python -c "from config import settings; print(settings.cors_origins, settings.env)"
   cd frontend && npm run build    # must succeed
   ```
2. **Deploy**:
   ```bash
   fly deploy -a insightai-api --remote-only
   vercel deploy --prod --force
   ```
3. **Post-deploy smoke** (run the full `fly logs` + curl chain from Diagnostic Procedure; expected:
   - `GET /health` → 200
   - `GET /ready` → 200 (after first deploy that includes the new endpoint; 503 on first 30s)
   - `GET /status` → `{"status": "no_index"}` initially
   - `POST /upload` with a small PDF → 200, then `/status` shows `vector_count > 0` within ~60s
   - `POST /queryHybrid` with a known fact from that PDF → 200 with citations
   - From the Vercel URL: open DevTools → Network → confirm no CORS or 404 errors on first query
4. **Rollback** if a step regresses:
   - Code rollback: `fly releases -a insightai-api` → `fly deploy -a insightai-api --image <previous-image>`.
   - Secret rollback: `fly secrets unset -a insightai-api KEY`.

---

## Out of Scope (explicitly)

- Adding multi-machine scaling (would need sticky session / shared cache).
- Migrating from `InMemoryStore` / Upstash to a managed Postgres-backed memory.
- Adding a frontend test suite, a backend test suite, or CI lint steps.
- Custom domain, SSL, monitoring (Sentry / Logtail), autoscaling rules.
- Switching the LLM provider away from OpenRouter.
- Re-architecting the chunking pipeline or adding GraphRAG.

These are tracked in README's "Future Improvements" and not part of the deploy-unblock goal.

---

## Open Questions

1. **What is the actual Fly app name?** (`insightai-api` is the value in `fly.toml`; confirm or supply the real one.)
2. **What is the actual Vercel production URL?** Needed for `CORS_ORIGINS`.
3. **Is the user willing to do `fly deploy` manually this once**, or must the GitHub Action be the only path? (Step 1 secrets must be set from a human machine; CI cannot create volumes or first-time secrets.)
4. **Is OpenRouter free model acceptable, or should we plan a swap to a paid tier / Gemini to remove the queue-latency failure mode (F4)?** Out of scope for "unblock deploy" but worth flagging.

If the user can supply (1) and (2) and confirm (3) = manual once, the plan is implementation-ready.
