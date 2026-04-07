# ManualIQ — Project Blueprint (CLAUDE.md)

> Offline RAG system for querying industrial/marine technical manuals. Engineering-grade cited answers. No cloud. No API keys.

---

## Architecture Overview

```
PDF Upload
    ↓
ingest.py       — Docling OCR → Markdown + figure captions (minicpm-v, optional)
    ↓
index.py        — Markdown → LanceDB vectors (nomic-embed-text, 768 dims)
                  Section-prefixed chunks: "[Section: A > B]\ntext"
                  Sibling links: prev_chunk_id / next_chunk_id
                  Lexical: chunks.jsonl (BM25)
    ↓
query.py        — Hybrid retrieval: vector ANN + BM25 + RRF merge
                  Load balancing (≥1 chunk per source)
                  Learned relevance boost (90-day decay, 2× cap)
    ↓
server.py       — FastAPI + SPA backend, streaming /api/v1/chat/stream (SSE)
                  Injects: memory summary + custom instructions + history
    ↓
Ollama          — deepseek-r1:8b (reasoning), nomic-embed-text (embedding)
```

**Two knowledge tiers:**
- `core_knowledge/` — shared domain fundamentals (11 equipment categories), indexed in `core_knowledge` LanceDB table, auto-ingested at startup via `preflight_core.py` + `watchdog`
- Per-notebook uploads — user PDFs, indexed in `manual_chunks` LanceDB table

---

## Key Scripts

| Script | Role |
|--------|------|
| `scripts/server.py` | **Primary entry point.** FastAPI backend + SPA. All `/api/*` endpoints. |
| `scripts/app.py` | Legacy Gradio UI. Shares `build_prompt()` + retrieval logic with server. |
| `scripts/common.py` | Shared utilities: `load_config()`, `resolve_path()`, `ensure_ollama_models()`, `fetch_ollama_models()` |
| `scripts/ingest.py` | PDF → Markdown via Docling. Vision captioning (optional). Sidecar `.ingest_timestamps.json`. |
| `scripts/index.py` | Markdown → LanceDB. `split_markdown()`, `index_markdown()`, `ChunkRecord` dataclass. |
| `scripts/query.py` | `retrieve()` (hybrid), `retrieve_vector()`, `retrieve_vectorless()`, BM25, fuzzy matching, boost application. |
| `scripts/notebook_map.py` | Per-notebook intelligence. `update_map()`, `get_boost_table()`, `regenerate_memory()`. |
| `scripts/preflight.py` | Validates environment (Ollama, models, deps). |
| `scripts/preflight_core.py` | Ingests + indexes `core_knowledge/` at startup. |
| `scripts/chat.py` | Terminal REPL for power users. |
| `scripts/populate_notebooks.py` | Bootstrap initial notebook registry. |

---

## Configuration (`config.yaml`)

All runtime settings live here. Loaded by `common.load_config()`.

```yaml
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  lancedb_dir: data/lancedb
  lancedb_table: manual_chunks
  notebooks_registry: data/notebooks.json
  chats_dir: data/chats
  maps_dir: data/maps            # Query maps + memory summaries

models:
  embedding: nomic-embed-text
  reasoning: deepseek-r1:8b
  vision: minicpm-v              # Optional, CPU-safe disabled by default

indexing:
  min_chunk_chars: 400
  max_chunk_chars: 1800
  overlap_chars: 200
  pdf_pages_per_batch: 15        # RAM-safe large PDF batching
  batch_overlap_pages: 2         # pages repeated between batches (preserves cross-boundary diagrams)

retrieval:
  mode: hybrid                   # vector | vectorless | hybrid
  top_k: 10
  load_balance: true

chat:
  show_thinking: false           # Strip <think> blocks
  history_turns: 6

runtime:
  ollama_num_thread: 12
  ollama_ctx_num: 16384
  ollama_num_predict: 4096
```

---

## Data Structures

### Notebook registry (`data/notebooks.json`)
```json
{
  "notebooks": [{
    "id": "nb_<uuid>",
    "name": "Main Engine Manual",
    "category": "1_propulsion_main_machinery",
    "custom_prompt": "Respond in SI units.",
    "sources": ["ME_manual.md", "core"],
    "core_books": ["1_propulsion_main_machinery"]
  }]
}
```

### ChunkRecord (`index.py`)
```python
@dataclass
class ChunkRecord:
    id: str               # UUID
    text: str
    source: str           # Filename
    section_path: str     # "Engine Room > Fuel System > Pump"
    headers: dict         # {h1, h2, h3, h4}
    notebook_id: str
    chunk_type: str       # "text" | "figure" | "table"
    prev_chunk_id: str    # Sibling link
    next_chunk_id: str    # Sibling link
```

### LanceDB tables
- **`manual_chunks`** — per-notebook user PDF chunks (768-dim vectors)
- **`core_knowledge`** — shared domain knowledge chunks, filtered by `equipment_category`

### Query map (`data/maps/{nb_id}_query_map.json`)
Tracks chunk usage counts + last-seen timestamps. Drives learned boost.

### Notebook memory (`data/maps/{nb_id}_memory.md`)
LLM-generated natural language summary of what the notebook covers well. Regenerated every 20 queries. Injected into system prompt.

### Chat history (`data/chats/{nb_id}.jsonl`)
Append-only JSONL. One line per turn with `role_user`, `role_assistant`, `citations`, `ts`.

---

## Key Algorithms

### Hybrid Retrieval (`query.py`)
1. Embed query → ANN search in LanceDB
2. BM25 on `chunks.jsonl` with fuzzy typo tolerance (`_fuzzy_match`, ≥4 char terms)
3. Reciprocal Rank Fusion (RRF) to merge ranked lists
4. Learned boost: `score *= min(2.0, 1 + log(usage)) × exp(-days/90)`
5. Load balance: guarantee ≥1 chunk per source, fill rest by rank

### Abbreviation Expansion (`server.py`)
Built-in 21-term marine map (ME→main engine, FO→fuel oil, TC→turbocharger, etc.) + user overrides from `data/abbreviations.json`. Both original and expanded terms sent to retrieval.

### THINK-GATE (`server.py`)
Simple factual lookups skip `<think>` reasoning step. Detected by query pattern heuristic. Saves ~3–5s per simple question.

### PARENT-CHUNK Expansion (`server.py`)
Top-N retrieved chunks expand to include their siblings (`prev_chunk_id` / `next_chunk_id`) for better context continuity.

### Chat turn data flow
```
/api/v1/chat/stream
  → query.retrieve()          — hybrid retrieval + boost
  → load history (last 6)     — data/chats/{nb_id}.jsonl
  → load memory               — data/maps/{nb_id}_memory.md
  → build_prompt()            — inject context + instructions + history
  → Ollama stream (SSE)       — deepseek-r1:8b
  → parse [Chunk N] citations
  → persist turn + update map
  → every 20 queries: regenerate_memory()
```

---

## Frontend (`static/index.html`)

Pure HTML/CSS/JS SPA — no framework, no build step. Dark theme.

- Fetch API for `/api/*`, EventSource for streaming SSE
- rAF-throttled scroll (prevents DOM janks)
- Incremental DOM append (not innerHTML replace)
- 262px sidebar: notebook list, source manager, upload, 🧠 intelligence panel
- Copy-to-clipboard per answer, citation panel

---

## Core Knowledge Structure

`core_knowledge/fundamentals/{category}/{subcategory}/{doc}.md`

11 categories (directory names are canonical):
1. `1_propulsion_main_machinery`
2. `2_fuel_oil_purification`
3. `3_pumps_piping_fluid`
4. `4_electrical_power_distribution`
5. `5_boiler_steam`
6. `6_compressed_air_gas`
7. `7_cooling_fresh_water_refrigeration`
8. `8_waste_pollution_control`
9. `9_deck_machinery_cargo_handling`
10. `10_bridge_navigation_equipment`
11. `11_firefighting_safety_lifesaving`

`watchdog` monitors this directory. New/modified `.md` files trigger re-ingest automatically.

---

## Thread Safety

- Query map updates: `_MAP_LOCK` (notebook_map.py)
- Chat persistence: `_lock` (server.py)
- Chunk cache: per-notebook LRU, invalidated on upload
- Boost cache: TTL, invalidated on map update

---

## Running the Project

```bash
# Prerequisites
ollama serve                          # Must be running on port 11434
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text

# Start server
python scripts/server.py              # http://127.0.0.1:7860
python scripts/server.py --port 8080  # custom port

# Validate environment
python scripts/preflight.py

# Tests
pytest tests/test_core.py -v                          # Unit (no Ollama needed)
MANUALIQ_INTEGRATION=1 pytest tests/test_search.py    # Integration (needs Ollama)
```

---

## CI/CD

`.github/workflows/ci.yml` — runs on push to `main` / `feature/*`, PR to `main`:
1. `ruff check` (E9, F63, F7, F82 errors fail)
2. `pytest tests/` (unit tests only)

---

## Performance Tuning

Priority order for speed improvements:

| Setting | Impact |
|---------|--------|
| `OLLAMA_KEEP_ALIVE=60m` | Eliminates 5–15s cold-load per query |
| `ollama_ctx_num: 8192` (down from 16384) | Faster prefill, less KV RAM |
| `OLLAMA_FLASH_ATTENTION=1` | ~1–2 GB KV RAM savings |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Reduced memory pressure |
| Windows High Performance power plan | Prevents CPU throttle |

**Model alternatives (faster, similar quality):**
- `phi4-mini` — 18–28 t/s, best MATH-500 score
- `qwen2.5:7b` — 14–20 t/s, no reasoning
- `deepseek-r1:8b` (current) — 8–14 t/s, best reasoning

---

## Pending Upgrades (priority order)

1. **OVERLAP-400** — chunk overlap 200→400 chars (requires full reindex)
2. **PARALLEL-RETRIEVE** — concurrent BM25 + vector retrieval
3. **EMBED-RERANK** — re-rank candidates by embedding similarity
4. **MEMORY-REGEN-LITE** — use smaller model for periodic memory summaries

---

## Roadmap

| Phase | Status | Theme |
|-------|--------|-------|
| 3 | ✅ | Persistence & multi-session |
| 3.5 | ✅ | Notebook intelligence (query maps, boost, memory) |
| 4 | Planned | REST API + integration layer |
| 5 | Planned | Multi-user + RBAC |
| 6 | Planned | Retrieval quality (reranking, decomposition) |
| 7 | Planned | Document format expansion (XLSX, PPTX, audio) |
| 8 | Planned | Agentic capabilities (PMS, alarms, checklists) |
| 9 | Planned | UX polish (global search, PDF preview, mobile) |

---

## Branches

- `main` — production-ready, all phases merged
- `feature/upgrades` — stale (fully merged into main)
- `feature/phase1-2-core-knowledge` — stale (fully merged into main)
- `feature/phase3-persistence-intelligence` — stale (fully merged into main)
