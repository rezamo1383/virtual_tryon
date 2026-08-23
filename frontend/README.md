# Vision Studio demo frontend

A presentation-focused Streamlit client for the existing tenant-aware FastAPI
backend. The frontend contains no prompt, model, preprocessing, or generation
logic. Wallpaper keeps the generic synchronous route. Clothing submits to the
asynchronous Try-On route and follows the returned job through SSE. The tenant
API key determines backend routing and is reused for status and result access.

## Features

- Product dashboard with dedicated Clothing and Wallpaper studios
- Multi-image clothing looks with a selected garment or accessory per reference
- Immediate drag-and-drop previews and built-in examples
- Light and dark appearance modes
- Live waiting, uploading, analyzing, generating, and completion feedback
- Full-resolution result preview, click-to-zoom, download, and comparison slider
- Session-local recent job history and one-click reset
- Sanitized client errors with no provider details or stack traces
- Responsive premium card layout suitable for client meetings

## Clothing request lifecycle

One click creates exactly one backend generation job:

```text
POST /api/v1/tryon
  -> HTTP 202 {"job_id":"job_xxx"}

GET /api/v1/jobs/{job_id}/events
  -> queued / running / completed

GET /api/v1/results/{job_id}/image
  -> final image bytes and MIME type
```

The Streamlit client uses `httpx.Client.stream()` for SSE. Status events update
the existing progress UI, while `: keep-alive` comments are ignored. The stream
is not replaced with job polling. A final image is requested exactly once after
`completed` or `completed_with_failures`; failed and rejected jobs do not trigger
an image request. The same clothing tenant API key is sent to all three
endpoints. Wallpaper retains its existing synchronous generic flow.

## Install

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-frontend.txt
```

## Run

Start the existing backend in the first terminal:

```powershell
cd F:\gold-list2\virtual_tryon
uvicorn api:app --host 127.0.0.1 --port 8000
```

Start Streamlit in a second terminal. Local development defaults to
`http://127.0.0.1:8000` only when `APP_ENV=development` (the non-container
default). Setting `API_BASE_URL` explicitly is recommended. Run from the
frontend directory so Streamlit loads its theme and optional secrets configuration:

```powershell
cd F:\gold-list2\virtual_tryon\frontend
$env:API_BASE_URL = "http://127.0.0.1:8000"
streamlit run app.py --server.port 8501
```

Alternatively, copy `.streamlit/secrets.toml.example` to the Git-ignored
`.streamlit/secrets.toml` and set `API_BASE_URL` there for local development.

Open `http://127.0.0.1:8501`.

## Configuration

Runtime values are read from environment variables first and then from
`.streamlit/secrets.toml`:

```text
API_BASE_URL
CLOTHING_API_KEY
WALLPAPER_API_KEY
COMPANY_NAME
API_TIMEOUT_SECONDS
MAX_HISTORY_ITEMS
```

Configuration priority is Environment Variable, Streamlit Secrets, then the
local-only development fallback. `API_BASE_URL` must be an absolute HTTP(S) URL.
The frontend Docker image sets `APP_ENV=production`, so the localhost fallback is
disabled in containers and a missing `API_BASE_URL` fails with a configuration
error instead of connecting to the container's loopback interface.

For a standalone Docker deployment, pass the backend container DNS name at
runtime; it is not stored in the frontend image:

```bash
docker run --name virtual-tryon-frontend \
  --network virtual-tryon-network \
  -p 8501:8501 \
  -e API_BASE_URL=http://virtual-tryon-backend:8000 \
  cr.samiansoft.com/virtual-tryon-frontend:latest
```

The local `secrets.toml` is ignored by Git. Only tenant access keys belong in
this frontend file. Provider credentials such as `GAPGPT_API_KEY` must remain
in the backend `.env` and are never exposed to Streamlit or the browser.

When rotating a tenant key, replace the raw value in frontend secrets and its
SHA-256 digest in `config/tenants.json`. The UI never sends `task_type` or a
tenant ID.

## Structure

```text
frontend/
├── app.py
├── assets/
├── components/
│   ├── media.py
│   ├── progress.py
│   ├── results.py
│   ├── sidebar.py
│   └── theme.py
├── config/
│   └── settings.py
├── pages/
│   ├── home.py
│   ├── clothing.py
│   ├── wallpaper.py
│   └── common.py
├── services/
│   ├── api_client.py
│   └── history.py
└── tests/
```

Recent jobs intentionally live only in the active Streamlit session. Clearing
history does not delete backend jobs or generated files.
