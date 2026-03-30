# Execution Checklist

## Current Workspace State
- [x] Created project scaffold and data directories
- [x] Config-driven scripts for ingestion / indexing / query / chat / app
- [x] `preflight.py` validator for local setup checks
- [x] `ingest.py` — PDF → Markdown via Docling (single or batch)
- [x] `index.py` — Markdown → LanceDB (multi-source, upsert)
- [x] `query.py` — single-shot CLI query (backward compat)
- [x] `chat.py` — terminal REPL with conversation history
- [x] `app.py` — Gradio web UI (primary entry point)
- [ ] Python packages installed into `.venv`
- [ ] Ollama models pulled locally (`deepseek-r1:8b`, `nomic-embed-text`)

---

## One-Time Setup

### 1. Create and activate virtual environment
```powershell
cd "D:\MASTER_PROJECTS\Offline Notebook LLM"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install all Python dependencies
```powershell
pip install -r requirements.txt
```
This installs: docling, gradio, lancedb, langchain, ollama, pydantic, pyyaml

### 3. Pull Ollama models (one-time — then fully offline)
Run in a **separate terminal**:
```powershell
ollama serve
```
Then in the main terminal:
```powershell
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

---

## Validate Setup
```powershell
python scripts/preflight.py
```
All checks should show `[OK]`. Fix any `[FAIL]` before proceeding.

---

## Option A — Web UI (Recommended)

### 1. Start Ollama (if not already running)
```powershell
ollama serve      # separate terminal
```

### 2. Launch the app
```powershell
python scripts/app.py
```
Opens `http://127.0.0.1:7860` in your browser automatically.

### 3. Use the UI
- **Upload PDFs** via the "Upload Manual(s)" button on the left panel
  - Each PDF is auto-ingested (Docling) and indexed (LanceDB) on upload
  - Processing status shown below the upload button
- **Chat** by typing questions in the right panel
  - Answers cite `[Chunk N]` from the manual
  - Retrieved chunks shown in the "Citations" box below chat
- **Remove a source** using the dropdown + "Remove Selected Source" button

---

## Option B — Terminal REPL (Power Users)

### Pre-ingest PDFs (if not using the web UI)
```powershell
python scripts/ingest.py --all         # convert all PDFs in data/raw/
python scripts/index.py                # chunk + index all Markdowns
python scripts/preflight.py --require-index
```

### Start terminal chat
```powershell
python scripts/chat.py
```
Commands inside REPL: `/chunks`  `/sources`  `/clear`  `exit`

---

## Option C — Single-Shot Query (Scripting / Testing)
```powershell
python scripts/query.py "How to troubleshoot low scavenge air pressure?" --retrieval-mode hybrid
```

---

## Adding a New Manual Later (Incremental)
```powershell
# Option 1 — via the web UI (easiest): just upload in the browser
# Option 2 — via CLI:
python scripts/ingest.py --input-pdf "data/raw/new_manual.pdf"
python scripts/index.py    # rebuilds full index (preserves all sources)
```

---

## Air-Gap Offline Test
1. Ensure `ollama serve` is running locally
2. Disable all network adapters
3. Run: `python scripts/preflight.py --require-index`
4. Run: `python scripts/app.py` — upload and chat should work with no network
5. Pass condition: answers returned with `[Chunk N]` citations, no errors

---

## Performance Tuning (32 GB RAM)

Default settings are tuned for this machine. If responses are slow:

| Setting | Default | Reduce to |
|---|---|---|
| `retrieval.top_k` | 6 | 4 |
| `runtime.ollama_ctx_num` | 8192 | 4096 |
| `indexing.max_chunk_chars` | 1800 | 1200 |
| `runtime.ollama_num_thread` | 12 | 8 |
