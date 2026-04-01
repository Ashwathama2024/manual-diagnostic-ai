# ManualIQ — Offline AI for Technical Manuals

> Ask questions against your industrial equipment manuals. Get cited, engineering-grade answers.
> Fully offline. No cloud. No API keys. Runs on AMD or NVIDIA GPU.

ManualIQ is a local Retrieval-Augmented Generation (RAG) system built for engineers and technicians who need fast, reliable answers from dense technical documentation — service manuals, maintenance guides, wiring diagrams, and more. Every answer is grounded in your documents and cites the exact chunk it came from.

---

## What's New — Phase 3 / 3.5 (Apr 2026)

**Persistence & Multi-Session**
- 💾 Chat history saved to disk — survives browser refresh, replayed on notebook select
- ⚙ Per-notebook custom instructions — prepended to every prompt, auto-saved
- ⬇ Export full conversation as Markdown with one click
- 🗓 Source timestamps — "Last updated" shown next to each uploaded document

**Notebook Intelligence Layer**
- 🗺 Per-notebook query map — tracks which chunks and sections actually answer questions
- 📈 Learned relevance boost — frequently-cited chunks get a rank nudge on future queries (2× cap, 90-day decay)
- 🧠 Notebook memory summary — LLM auto-generates a profile every 20 queries; injected into system prompt; viewable in sidebar
- ⚖ Source load balancing — every selected source guaranteed ≥1 chunk in context; large manuals can no longer silence small bulletins

**UX Fixes**
- 🔧 Chat scroll freeze fixed — incremental DOM append instead of full text replace per token
- ⧉ Copy button on every assistant answer

---

## Features

| Capability | Detail |
|---|---|
| **100% Offline** | Ollama for local LLM inference — air-gap capable after model download |
| **Docling OCR Pipeline** | Handles scanned/image-heavy PDFs that other tools fail on |
| **Hybrid Retrieval** | Vector (semantic) + BM25 lexical search merged for best precision |
| **Multi-Notebook** | Isolate different equipment manuals into separate workspaces |
| **DeepSeek-R1 Reasoning** | Chain-of-thought model with live `🧠 Reasoning…` indicator |
| **Vision Captioning** | Optional: extract & caption diagrams via multimodal model |
| **AMD GPU Support** | Vulkan compute via `OLLAMA_VULKAN=1` — works on AMD Radeon iGPU/dGPU |
| **FastAPI SPA** | Custom dark-mode single-page app — real-time SSE streaming; no framework dependency |
| **Anti-Hallucination** | Every answer must cite `[Chunk N]` or return a fixed refusal string |
| **Skip Re-indexing** | Already-embedded documents are detected and skipped automatically |
| **Typed Chunks** | Chunks tagged as `text` / `figure` / `table` for richer retrieval |
| **Persistent Chat History** | Chat turns saved to `data/chats/{nb_id}.jsonl`; replayed on notebook select; survives browser refresh |
| **Per-Notebook Instructions** | Custom system prompt per notebook (⚙ gear panel); prepended to every prompt; auto-saved |
| **Export Conversation** | One-click Markdown download of full chat session (`⬇ Export` in title bar) |
| **Ingest Timestamps** | "Last updated" date shown next to each source in the sidebar |
| **Notebook Intelligence** | Per-notebook query map tracks which chunks and sections answer questions; drives learned boost |
| **Learned Relevance Boost** | Frequently-cited chunks get a rank nudge on future queries (max 2×, 90-day decay) |
| **Notebook Memory Summary** | LLM auto-generates a 2–4 sentence profile every 20 queries; injected into system prompt; shown in `🧠` sidebar panel |
| **Source Load Balancing** | Every selected source guaranteed ≥1 chunk in context — prevents large manuals silencing small bulletins |
| **Smooth Streaming UX** | Incremental DOM append (no full replace per token); rAF-throttled scroll; per-answer copy button (⧉) |

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ingest.py  —  Docling Python API                    │
│  • do_ocr=True  (RapidOCR for scanned pages)         │
│  • Table structure extraction → markdown tables      │
│  • Figure extraction → minicpm-v captions (optional) │
│  Output: {notebook_id}/{filename}.md                 │
│  Sidecar: .ingest_timestamps.json (per source)       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  index.py  —  Chunking & Embedding                   │
│  • LangChain MarkdownHeaderTextSplitter              │
│  • Section-prefix baked into vectors                 │
│    "[Section: A > B]\n{chunk text}"                  │
│  • nomic-embed-text via Ollama                       │
│  • Stored in LanceDB (serverless)                    │
│  • Lexical chunks saved to chunks.jsonl              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  query.py  —  Hybrid Retrieval                       │
│  • Vector search (LanceDB ANN)                       │
│  • BM25 lexical search (chunks.jsonl)                │
│  • Fetch 2× pool → load-balance (≥1 per source)     │
│  • Apply learned relevance boost (notebook_map.py)   │
│  • Trim to top-k                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  app.py / server.py  —  Prompt Builder               │
│  • Inject notebook memory summary (if available)     │
│  • Inject per-notebook custom instructions           │
│  • Build [Chunk N] context + history block           │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  DeepSeek-R1:8b via Ollama                           │
│  • System prompt enforces citation + no hallucination│
│  • <think> blocks stripped; 🧠 indicator shown live  │
│  • Streamed token-by-token to UI                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  notebook_map.py  —  Notebook Intelligence           │
│  • Append turn to data/chats/{nb_id}.jsonl           │
│  • Update data/maps/{nb_id}_query_map.json           │
│    – chunk_usage, section_usage, query_log           │
│  • Every 20 queries: regenerate memory summary       │
│    data/maps/{nb_id}_memory.md                       │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running

Pull the required models:
```bash
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

For AMD GPU acceleration (optional but highly recommended):
```bash
# Windows — set as a System Environment Variable:
OLLAMA_VULKAN=1
# Then restart Ollama
```

### 2. Install

```bash
git clone https://github.com/Ashwathama2024/manual-diagnostic-ai.git
cd manual-diagnostic-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch

**Option A — FastAPI SPA (recommended)**
```bash
python scripts/server.py --port 7860
# Opens http://127.0.0.1:7860
```

**Option B — Gradio Web UI (legacy)**
```bash
python scripts/app.py
# Opens http://127.0.0.1:7860
```

**Option C — Terminal REPL (power users)**
```bash
python scripts/chat.py --retrieval-mode hybrid
```

---

## UI Overview (FastAPI SPA)

```
┌──────────────────────────┬────────────────────────────────────────────┐
│  SIDEBAR                 │  TITLE BAR                                 │
│  ─────────────────────   │  📓 Main Engine  ·  3 sources · 412 chunks │
│  [Select notebook ▼]     │  [⬇ Export]  [✕ Clear]                    │
│  [+ New]                 ├────────────────────────────────────────────┤
│                          │  MESSAGES                                  │
│  SOURCES                 │                                            │
│  ─────────────────────   │  You: What is the FO pump pressure?        │
│  ☑ 📄 ME_manual.md       │                                            │
│    Last updated: Apr 1   │  ManualIQ: Based on [Chunk 3] the normal   │
│  ☑ 📄 FO_system.md       │  operating pressure is 8–10 bar…  [⧉ Copy]│
│  ☑ Core: propulsion      │                                            │
│  [⬆ Upload PDF / DOCX]   │  [Type your question…]          [Send ➤]   │
│  [🗑 Remove Selected]    ├────────────────────────────────────────────┤
│                          │  CITATIONS                                 │
│  ⚙ NOTEBOOK SETTINGS     │  [1] FO_system.md › Pressure Specs        │
│  ┌──────────────────┐    │  [2] ME_manual.md › Fuel System            │
│  │ Custom prompt…   │    └────────────────────────────────────────────┘
│  └──────────────────┘
│
│  🧠 NOTEBOOK INTELLIGENCE
│  "Covers MAN B&W 6S60MC-C.
│   Strong: fuel injection,
│   turbocharger maintenance."
│  [↺ Regenerate]
│
│  Docs: 3  ·  Chunks: 412
└──────────────────────────
```

---

## Configuration (`config.yaml`)

```yaml
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  lancedb_dir: data/lancedb
  lancedb_table: manual_chunks
  notebooks_registry: data/notebooks.json
  chats_dir: data/chats       # per-notebook chat history JSONL
  maps_dir: data/maps         # per-notebook query maps + memory summaries

models:
  embedding: nomic-embed-text      # Ollama embedding model
  reasoning: deepseek-r1:8b        # Reasoning/chat LLM
  vision: minicpm-v                # Vision model for diagram captions

indexing:
  min_chunk_chars: 400
  max_chunk_chars: 1800
  overlap_chars: 200
  batch_size: 16

retrieval:
  mode: hybrid                     # vector | vectorless | hybrid
  top_k: 6
  load_balance: true               # guarantee ≥1 chunk per source (default on)

chat:
  show_thinking: false             # Show DeepSeek <think> blocks
  history_turns: 6

vision:
  enabled: false                   # Enable diagram captioning (GPU recommended)
  timeout: 180                     # Seconds per image
  max_per_doc: 10

runtime:
  ollama_num_thread: 12
  ollama_ctx_num: 8192
```

---

## Scripts Reference

| Script | Purpose | Usage |
|---|---|---|
| `scripts/server.py` | FastAPI backend + SPA (primary) | `python scripts/server.py` |
| `scripts/app.py` | Gradio web UI (legacy) + shared helpers | `python scripts/app.py` |
| `scripts/notebook_map.py` | Query map, learned boost, memory summary | (library — imported by server.py) |
| `scripts/ingest.py` | PDF → Markdown | `python scripts/ingest.py file.pdf` |
| `scripts/index.py` | Markdown → LanceDB | `python scripts/index.py` |
| `scripts/query.py` | One-shot retrieval + answer | `python scripts/query.py "question"` |
| `scripts/chat.py` | Terminal REPL | `python scripts/chat.py` |
| `scripts/preflight.py` | Validate environment | `python scripts/preflight.py` |
| `scripts/common.py` | Shared utilities | (library — not run directly) |

---

## Data Directory Layout

```
data/
├── notebooks.json              # Notebook registry (id, name, category, custom_prompt)
├── raw/{nb_id}/                # Original PDFs/DOCX per notebook
├── processed/{nb_id}/
│   ├── *.md                    # Docling-converted markdown
│   ├── chunks.jsonl            # BM25 lexical index
│   └── .ingest_timestamps.json # Per-source ingest dates
├── chats/{nb_id}.jsonl         # Chat history (one turn per line)
├── maps/
│   ├── {nb_id}_query_map.json  # Usage tracking + boost table
│   └── {nb_id}_memory.md      # LLM-generated notebook summary
└── lancedb/
    ├── manual_chunks.lance/    # Notebook chunk vectors
    └── core_knowledge.lance/   # Core knowledge base vectors
```

---

## Pipeline: How a PDF Becomes Searchable

1. **Upload** — PDF dropped into the UI or placed in `data/raw/`
2. **Ingest** (`ingest.py`) — Docling converts to structured Markdown with:
   - Full OCR on scanned/image pages (RapidOCR)
   - Tables preserved as Markdown tables
   - Figures optionally captioned by multimodal model
   - Timestamp written to `.ingest_timestamps.json`
3. **Chunk** (`index.py`) — LangChain splits on Markdown headers; each chunk records its `section_path` (e.g., `Engine Room > Fuel System > Pump Specs`)
4. **Embed** (`index.py`) — `nomic-embed-text` embeds `"[Section: X > Y]\nchunk text"` so section context is baked into the vector
5. **Store** — Vectors go to LanceDB; raw text goes to `chunks.jsonl` for lexical search
6. **Query** (`query.py` + `server.py`) — BM25 + vector results fetched (2× pool) → load-balanced → learned boost applied → top-k returned
7. **Reason** — DeepSeek-R1 receives chunks + memory summary + custom instructions + history; streams cited answer
8. **Learn** (`notebook_map.py`) — Turn written to chat JSONL; query map updated with chunk usage; memory regenerated every 20 queries

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 32 GB |
| CPU | 4 cores | 12+ cores |
| GPU | None (CPU-only) | AMD Radeon (4GB+ VRAM via BIOS UMA) or NVIDIA 8GB+ |
| Storage | 5 GB (models) | 20 GB (models + docs) |

**AMD iGPU tip**: If your laptop has an AMD Radeon integrated GPU with low dedicated VRAM, increase the UMA Frame Buffer Size in BIOS to 4–8 GB. Set `OLLAMA_VULKAN=1` in System Environment Variables. This provides 10–15× faster inference vs CPU.

---

## Retrieval Modes

| Mode | How it works | Best for |
|---|---|---|
| `vector` | Semantic similarity via nomic-embed-text | Conceptual / paraphrased questions |
| `vectorless` | BM25 keyword scoring on chunks.jsonl | Exact part numbers, codes, spec lookups |
| `hybrid` | Both merged + deduplicated + load-balanced (default) | All-around best results |

---

## FAQ

**Q: Does this need internet?**
No. After pulling Ollama models once, everything runs offline. No telemetry, no API calls.

**Q: What PDF types work best?**
Both text-based and scanned PDFs work. Docling's OCR handles image-heavy manuals that tools like PyMuPDF can't extract from.

**Q: Can I use a different LLM?**
Yes — change `models.reasoning` in `config.yaml` to any model you've pulled via `ollama pull`. Tested with `llama3.2:3b`, `qwen2.5:7b`, and `deepseek-r1:8b`.

**Q: How do I enable vision diagram captioning?**
Set `vision.enabled: true` in `config.yaml`. Requires `minicpm-v` pulled via Ollama. Strongly recommended to have GPU — CPU inference is ~3 min/image.

**Q: Does chat history persist after a browser refresh?**
Yes (Phase 3). Every turn is saved to `data/chats/{nb_id}.jsonl` and replayed automatically when you select a notebook.

**Q: What is the Notebook Intelligence panel?**
The 🧠 panel shows a natural-language summary of what the notebook is good at answering. It is generated by the LLM after every 20 queries and injected into the system prompt so answers become more contextually aware over time.

**Q: How does load balancing work?**
When you have multiple sources selected, the retriever fetches a 2× candidate pool, then guarantees at least one chunk from every source before trimming to `top_k`. Set `retrieval.load_balance: false` in `config.yaml` to disable.

---

## License

MIT — see [LICENSE](LICENSE)
