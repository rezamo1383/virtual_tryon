# Vision Studio demo frontend

A presentation-focused Streamlit client for the existing tenant-aware FastAPI
backend. The frontend contains no prompt, model, preprocessing, or generation
logic. Both products submit the same domain-neutral multipart request to
`POST /api/v1/generate`; the tenant API key determines backend routing.

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

Start Streamlit in a second terminal. Run it from the frontend directory so
Streamlit loads the included theme and secrets configuration:

```powershell
cd F:\gold-list2\virtual_tryon\frontend
streamlit run app.py --server.port 8501
```

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
