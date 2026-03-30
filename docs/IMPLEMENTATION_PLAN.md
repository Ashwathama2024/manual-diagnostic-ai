# Offline Manual Assistant — Implementation Plan

## Target Machine Profile
- RAM: 32 GB
- CPU: 16 logical processors
- OS: Windows 11 Pro
- Network: Air-gap capable (offline after initial model download)

---

## Architecture

```
User Browser / Terminal
       │
       ▼
 scripts/app.py          ← Gradio web UI (primary entry point)
 scripts/chat.py         ← Terminal REPL (power-user fallback)
       │
       ├── scripts/ingest.py   ← PDF → Markdown (Docling)
       ├── scripts/index.py    ← Markdown → LanceDB (nomic-embed-text)
       └── scripts/query.py    ← Retrieval (vector / lexical / hybrid)
                                  + DeepSeek-R1 reasoning (Ollama)
```

### Stack

| Layer | Tool | Why |
|---|---|---|
| Ingestion | Docling | Best table/diagram preservation from PDFs |
| Orchestration | LangChain text splitters | Section-aware chunking |
| Vector DB | LanceDB | Serverless, fast on local SSD, no server process |
| LLM Engine | Ollama | Easiest GGUF model runner on CPU/GPU |
| Reasoning Model | deepseek-r1:8b | High reasoning capability, 8B fits in 32 GB |
| Embedding Model | nomic-embed-text | 270 MB, 8k token context, fast |
| Web UI | Gradio | Native chat + file upload, offline, pip-installable |

---

## Repo Structure

```
scripts/
  app.py          — Gradio web app (PRIMARY ENTRY POINT)
  chat.py         — Terminal REPL
  ingest.py       — PDF → Markdown conversion (Docling)
  index.py        — Chunking + LanceDB indexing
  query.py        — Retrieval + reasoning (single-shot CLI)
  preflight.py    — Environment validation
  common.py       — Shared utilities

data/
  raw/            — Source PDFs (user-supplied)
  processed/      — Per-PDF Markdown files (auto-generated)
    chunks.jsonl  — All chunks merged (for lexical retrieval)
  lancedb/        — Vector index
    manual_chunks.lance/

config.yaml       — All settings (models, paths, runtime)
requirements.txt  — Python dependencies
```

---

## Anti-Hallucination Design

Every response from the LLM is constrained by a strict system prompt:

1. **Mandatory chunk citations**: Every claim must cite `[Chunk N]` inline
2. **Hard refusal string**: If not in chunks → exact fixed response, no guessing
3. **Verbatim quoting**: Numbers, part codes, procedures quoted word-for-word
4. **No synthesis**: Cannot infer conclusions beyond what is explicitly written
5. **Fresh retrieval per turn**: AI answers from prior turns never feed back into retrieval (breaks hallucination loops)

---

## Multi-Source Design

- All PDFs ingested to `data/processed/{stem}.md`
- All Markdowns chunked and merged into one LanceDB table with a `source` field per row
- Upload → ingest → index pipeline is incremental: adding a new PDF upserts only its rows (preserves all others)
- Removing a source deletes its rows from LanceDB and its `.md` file

---

## DeepSeek-R1 Thinking Token Handling

DeepSeek-R1 outputs `<think>...</think>` blocks before its final answer. These are stripped by default:

```python
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_thinking(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
```

To see the reasoning chain: `python scripts/app.py --show-thinking`

---

## Day 0 Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Separate terminal:
ollama serve

# Pull models (one-time):
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

## Daily Use

```powershell
ollama serve          # if not running
python scripts/app.py # opens http://127.0.0.1:7860
```

---

## Performance Defaults for 32 GB

| Setting | Value | Notes |
|---|---|---|
| `runtime.ollama_num_thread` | 12 | Leave 4 threads for OS |
| `runtime.ollama_ctx_num` | 8192 | ~4k tokens input + 4k output |
| `retrieval.top_k` | 6 | ~2700 tokens of context |
| `indexing.max_chunk_chars` | 1800 | ~450 tokens per chunk |
| `indexing.batch_size` | 16 | Embedding batch size |

### If latency is high, reduce in this order:
1. `retrieval.top_k`: 6 → 4
2. `runtime.ollama_ctx_num`: 8192 → 4096
3. `indexing.max_chunk_chars`: 1800 → 1200
