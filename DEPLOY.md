# InsightAI — Deploy Guide (Fly.io + Vercel)

## 1. Chuẩn bị

```bash
# Fly CLI
curl -L https://fly.io/install.sh | sh
fly auth login

# Vercel CLI
npm i -g vercel
vercel login
```

## 2. Backend — Fly.io

```bash
cd backend

# Tạo app (lần đầu)
fly launch --no-deploy --copy-config

# Tạo volume 2GB (lần đầu, region sin)
fly volumes create insightai_data --region sin --size 2 --yes

# Set secrets (thay xxx bằng giá trị thật)
fly secrets set \
  OPENROUTER_API_KEY=xxx \
  GOOGLE_API_KEY=xxx \
  CORS_ORIGINS="https://your-vercel-app.vercel.app" \
  UPSTASH_REDIS_REST_URL=xxx \
  UPSTASH_REDIS_REST_TOKEN=xxx

# Deploy
fly deploy --remote-only

# Kiểm tra
fly status
fly logs
curl https://insightai-api.fly.dev/health
curl https://insightai-api.fly.dev/status
```

**Lưu ý:**
- `VECTOR_DB_PATH=/data/vector_db` và `UPLOAD_DIR=/data/uploads` đã set trong `fly.toml` → persistence qua redeploy.
- Lần đầu boot sẽ tải `BAAI/bge-small-en-v1.5` (~120MB) vào `/data/hf_cache` → boot 60-90s là bình thường.
- Nếu đổi CORS: `fly secrets set CORS_ORIGINS="https://new-domain.vercel.app"`

## 3. Frontend — Vercel

```bash
cd frontend

# Lần đầu: link project
vercel link

# Set env trên Vercel Dashboard hoặc CLI
vercel env add VITE_API_BASE_URL production
# Nhập: https://insightai-api.fly.dev

# Deploy
vercel --prod
```

Hoặc set trong **Vercel Dashboard → Settings → Environment Variables**:
- `VITE_API_BASE_URL` = `https://insightai-api.fly.dev`

## 4. CI/CD (GitHub Actions)

Thêm secrets trong **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Giá trị |
|--------|---------|
| `FLY_API_TOKEN` | `fly tokens create deploy -x 999h` |
| `VERCEL_TOKEN` | `vercel token` (https://vercel.com/account/tokens) |
| `VERCEL_ORG_ID` | từ `.vercel/project.json` sau `vercel link` |
| `VERCEL_PROJECT_ID` | từ `.vercel/project.json` |

Push lên `main` sẽ tự deploy cả backend và frontend.

## 5. Backup & Restore vector_db

```bash
# Backup từ Fly volume về local
fly sftp get /data/vector_db/index.faiss ./vector_db_backup/index.faiss

# Hoặc dùng fly ssh
fly ssh console -C "tar czf - /data/vector_db" > vector_db.tar.gz
```

## 6. Troubleshooting

| Lỗi | Cách fix |
|-----|----------|
| `CORS_ORIGINS is required in production` | `fly secrets set CORS_ORIGINS="https://..."` |
| Boot timeout / healthcheck fail | Tăng `grace_period` trong `fly.toml` lên `120s` |
| `No module named 'faiss'` | Check `backend/Dockerfile` đã cài `faiss-cpu` |
| Frontend 404 sau refresh | Đã có `vercel.json` rewrites → kiểm tra `vercel logs` |

---

## 7. First-time Bootstrap Checklist (lần đầu deploy)

Chạy đúng thứ tự — bước 1-3 chỉ chạy MỘT LẦN, bước 4-5 lặp lại mỗi lần deploy.

```bash
# 0. Login
fly auth login
vercel login

# 1. Tạo Fly app + copy config (chỉ chạy lần đầu, KHÔNG chạy lại)
cd backend
fly launch --no-deploy --copy-config

# 2. Tạo volume 2GB (chỉ chạy lần đầu)
fly volumes create insightai_data -a insightai-api --region sin --size 2 --yes
fly volumes list -a insightai-api   # phải thấy attached

# 3. Set TẤT CẢ secrets (lần đầu phải tự làm, CI không tạo secrets)
fly secrets set -a insightai-api \
  CORS_ORIGINS="https://<your-vercel-app>.vercel.app" \
  CORS_ORIGINS_DEFAULT="https://<your-vercel-app>.vercel.app" \
  OPENROUTER_API_KEY=<key> \
  UPSTASH_REDIS_REST_URL=<url> \
  UPSTASH_REDIS_REST_TOKEN=<token>

# 4. Build + deploy
fly deploy -a insightai-api --remote-only
vercel deploy --prod --force

# 5. Verify (chạy ngay sau deploy, đợi 1-2 phút cho lần đầu)
curl -s https://insightai-api.fly.dev/         # status:ok + build info
curl -s https://insightai-api.fly.dev/health   # status:ok
curl -s https://insightai-api.fly.dev/ready    # ready:true
curl -s https://insightai-api.fly.dev/status   # status:no_index (chưa upload)
fly logs -a insightai-api --no-tail | head -20
```

## 8. Common Errors → Fix

| Triệu chứng (`fly logs` / HTTP) | Nguyên nhân | Fix |
|---|---|---|
| `RuntimeError: CORS_ORIGINS is required in production but is empty` | Chưa set secret CORS_ORIGINS | `fly secrets set -a insightai-api CORS_ORIGINS="https://<app>.vercel.app"` |
| `/ready` 503 với `reason: embeddings_not_loaded` (>3 phút) | HF download fail (offline / rate limit) | Check `fly logs \| grep -i huggingface`; thử `fly ssh console -a insightai-api -C "ls /data/hf_cache"`; re-deploy |
| `/ready` 503 forever, logs show `ModuleNotFoundError: No module named 'faiss'` | requirements.txt thiếu hoặc pip fail | `fly deploy -a insightai-api --remote-only` lại; check `pip install` output trong logs |
| Boot timeout / healthcheck fail | Grace period quá ngắn (cũ: 60s) | Đã bump lên 180s; nếu vẫn fail, sửa `grace_period = "300s"` trong `fly.toml` |
| Boot OK, `/status` luôn `no_index` sau khi upload | Volume chưa mount | `fly volumes list -a insightai-api` → nếu rỗng, chạy lại bước 2 |
| Index biến mất sau mỗi redeploy | Volume chưa được tạo | Tạo volume trước khi deploy lần đầu (bước 2) |
| Frontend `Network Error` tới `http://localhost:8000` | Vercel `VITE_API_BASE_URL` sai | `vercel env ls --prod`; nếu sai, `vercel env add VITE_API_BASE_URL production` rồi `vercel deploy --prod --force` |
| Frontend CORS error trong DevTools | `CORS_ORIGINS` không khớp domain Vercel | Set lại `CORS_ORIGINS` đúng domain (kể cả `www.` nếu có); redeploy |
| Chat hang > 30s không có token | OpenRouter free model bị queue | Đổi `OPENROUTER_MODEL_NAME` sang model trả phí; hoặc nâng timeout trong frontend |
| `429 Too Many Requests` từ OpenRouter | Free model rate limit | Đổi sang model trả phí, hoặc thêm backoff retry |
| 500 khi upload file lần thứ 2 | Race condition cũ giữa `/reset` và `/upload` | Đã fix bằng `threading.Lock`; redeploy bản mới |
| Session chat bị mất khi ai đó upload file | Cũ: `clear_index` xóa luôn session store | Đã fix; conversation store tách khỏi index reset |

## 9. Quick Health-Check Script (chạy sau mỗi deploy)

```bash
APP=insightai-api
echo "=== Fly status ==="
fly status -a $APP
echo "=== Health endpoints ==="
curl -fsS https://$APP.fly.dev/health | jq
curl -fsS https://$APP.fly.dev/ready  | jq
curl -fsS https://$APP.fly.dev/status | jq
echo "=== Recent log lines ==="
fly logs -a $APP --no-tail | tail -30
```

Nếu `ready=true` + `status=no_index` + log không có lỗi → backend OK, chỉ cần upload file là dùng được.
