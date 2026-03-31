# ManualIQ — Offline AI for Technical Manuals

> Ask questions against your industrial equipment manuals. Get cited, engineering-grade answers.
> Fully offline. No cloud. No API keys. Runs on AMD or NVIDIA GPU.

ManualIQ is a local Retrieval-Augmented Generation (RAG) system built for engineers and technicians who need fast, reliable answers from dense technical documentation — service manuals, maintenance guides, wiring diagrams, and more. Every answer is grounded in your documents and cites the exact chunk it came from.

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
| **Two UI Options** | Gradio web app **or** FastAPI + React SPA — choose your stack |
| **Anti-Hallucination** | Every answer must cite `[Chunk N]` or return a fixed refusal string |
| **Skip Re-indexing** | Already-embedded documents are detected and skipped automatically |
| **Typed Chunks** | Chunks tagged as `text` / `figure` / `table` for richer retrieval |

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
│  • Merge + deduplicate → top-k chunks                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  DeepSeek-R1:8b via Ollama                           │
│  • System prompt enforces citation + no hallucination│
│  • <think> blocks stripped; 🧠 indicator shown live  │
│  • Streamed token-by-token to UI                     │
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

**Option A — Gradio Web UI (recommended)**
```bash
python scripts/app.py
# Opens http://127.0.0.1:7860
```

**Option B — FastAPI + React SPA**
```bash
python scripts/server.py --port 7860
# Opens http://127.0.0.1:7860
```

**Option C — Terminal REPL (power users)**
```bash
python scripts/chat.py --retrieval-mode hybrid
```

---

## UI Overview

### Gradio (app.py)

```
┌──────────────────────┬────────────────────────────────────┐
│  Notebooks           │  Chat                              │
│  ─────────────────   │  ─────────────────────────────     │
│  [Select notebook ▼] │  [Conversation history]            │
│  [+ Create]          │                                    │
│                       │  [Type your question…]  [Send]    │
│  Sources             │  [Clear]                           │
│  ─────────────────   │  ──────────────────────────────    │
│  [Upload PDF]        │  Citations                         │
│  Status: ✓ indexed   │  [Chunk N] Section > Sub           │
│  ● manual.md         │  Score: 87%  Page: 14              │
│  [Remove ▼]          │                                    │
└──────────────────────┴────────────────────────────────────┘
```

---

## Configuration (`config.yaml`)

```yaml
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
| `scripts/app.py` | Gradio web UI (primary) | `python scripts/app.py` |
| `scripts/server.py` | FastAPI backend (SPA) | `python scripts/server.py` |
| `scripts/ingest.py` | PDF → Markdown | `python scripts/ingest.py file.pdf` |
| `scripts/index.py` | Markdown → LanceDB | `python scripts/index.py` |
| `scripts/query.py` | One-shot retrieval + answer | `python scripts/query.py "question"` |
| `scripts/chat.py` | Terminal REPL | `python scripts/chat.py` |
| `scripts/preflight.py` | Validate environment | `python scripts/preflight.py` |
| `scripts/common.py` | Shared utilities | (library — not run directly) |

---

## Pipeline: How a PDF Becomes Searchable

1. **Upload** — PDF dropped into the UI or placed in `data/raw/`
2. **Ingest** (`ingest.py`) — Docling converts to structured Markdown with:
   - Full OCR on scanned/image pages (RapidOCR)
   - Tables preserved as Markdown tables
   - Figures optionally captioned by multimodal model
3. **Chunk** (`index.py`) — LangChain splits on Markdown headers; each chunk records its `section_path` (e.g., `Engine Room > Fuel System > Pump Specs`)
4. **Embed** (`index.py`) — `nomic-embed-text` embeds `"[Section: X > Y]\nchunk text"` so section context is baked into the vector
5. **Store** — Vectors go to LanceDB; raw text goes to `chunks.jsonl` for lexical search
6. **Query** (`query.py`) — BM25 + vector results merged, top-k returned
7. **Reason** — DeepSeek-R1 receives chunks + system prompt; streams cited answer

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
| `hybrid` | Both merged + deduplicated (default) | All-around best results |

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



---

## License

MIT — see [LICENSE](LICENSE)
