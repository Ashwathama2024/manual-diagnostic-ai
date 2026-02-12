# Manual-RAG Diagnostic Assistant

**AI-powered equipment diagnosis from your technical manuals — runs 100% offline.**

Upload PDF manuals (text, tables, diagrams). Ask diagnostic questions. Get engineering-grade answers with precise section + page citations. No internet required. No data leaves your machine.

> **Last updated:** 2026-02-12 | **Version:** 2.1 — Vision-aware diagram processing

---

## What Makes This Different

- **Auto-pipeline** — Upload PDF → text + tables + diagrams extracted → embedded → stored. One click.
- **Vision-powered diagrams** — Ollama vision models (minicpm-v / LLaVA) describe engineering diagrams with component labels, connections, measurements — not just OCR gibberish
- **Spatial context preservation** — Diagrams carry their surrounding text (captions, paragraphs above/below) so relevance is never lost
- **Semantic chunking** — Every chunk knows its chapter, section, and subsection. Not just "Page 45" but "MAN B&W Manual > Chapter 3: Fuel System > 3.2.1 Injection Timing > Page 45"
- **Equipment isolation** — Each machine gets its own vector database. Main engine data never mixes with generator data.
- **Streaming chat** — Responses appear word-by-word as the LLM generates them
- **9 recommended models** — from Phi-3 (8GB RAM) to Qwen 2.5 72B (rivals GPT-4)
- **100% offline** after setup — no cloud APIs, no telemetry, no data leaks

---

## Architecture

```
Upload PDF ──> Doc Processor ──────────> ChromaDB ──> Diagnostic Chat
                 │                          │              │
                 ├─ Text (PyMuPDF)          │              ├─ User question
                 ├─ Tables (pdfplumber)     │              ├─ Vector search (top-k)
                 ├─ Images ──┐              │              ├─ Context + question → LLM
                 │   ├─ Save to disk        │              └─ Streaming answer
                 │   ├─ Bounding box        │                  + section citations
                 │   ├─ Surrounding text    │
                 │   └─ Vision model ───┐   │
                 │       (or OCR fallback)  │
                 ├─ Section detect          ├─ Equipment A collection
                 └─ Semantic chunk          ├─ Equipment B collection
                     with hierarchy         └─ Equipment C collection
```

### Image Processing Pipeline (v2.1)

```
PDF Image → Extract bytes (PyMuPDF) → Size check → Dedup (MD5 hash)
  → Get bounding box on page (get_image_rects)
  → Find surrounding text blocks (spatial proximity)
  → Save to ./images/{equipment_id}/{file}_page{N}_img{M}.png
  → Describe with Ollama vision model (minicpm-v/llava)
      ↓ fallback: Tesseract OCR → fallback: skip
  → Combine: [section context] + [surrounding text] + [vision description]
  → Embed as chunk with image_path metadata
```

**Key design decisions:**
- **Section-aware chunking** — regex-based heading detection (CHAPTER, numbered sections, ALL CAPS headers) builds a breadcrumb hierarchy for every chunk
- **Chunk prefixing** — each chunk is prefixed with its section path so the embedding captures topic context, not just raw text
- **Vision-first diagrams** — engineering diagrams are described by a vision LLM (components, connections, measurements), not just OCR'd
- **Surrounding text context** — text above/below diagrams is captured via bounding-box geometry, so "Fig 3.2 — Fuel injection pump" stays linked to its diagram
- **Equipment isolation** — separate ChromaDB collections per equipment type
- **BGE embeddings** — `BAAI/bge-small-en-v1.5` gives better retrieval accuracy than MiniLM with similar speed
- **Graceful fallbacks** — vision → OCR → skip. System never fails because of the image pipeline

---

## Recommended Models

### Text LLM (for diagnostic chat)

| Model | Command | RAM | Quality | Best For |
|-------|---------|-----|---------|----------|
| **Llama 3.3 8B** | `ollama pull llama3.3:8b` | 16 GB | Excellent | Best all-rounder |
| **Qwen 2.5 7B** | `ollama pull qwen2.5:7b` | 16 GB | Excellent | Technical reasoning |
| **DeepSeek R1 8B** | `ollama pull deepseek-r1:8b` | 16 GB | Excellent | Chain-of-thought diagnostics |
| **Gemma 2 9B** | `ollama pull gemma2:9b` | 16 GB | Excellent | Instruction following |
| **Command R 35B** | `ollama pull command-r:35b` | 32 GB | Excellent | Built for RAG + citations |
| **Llama 3.3 70B** | `ollama pull llama3.3:70b` | 48 GB | Best | Maximum quality |
| **Qwen 2.5 72B** | `ollama pull qwen2.5:72b` | 48 GB | Best | Rivals GPT-4 |
| Phi-3 3.8B | `ollama pull phi3:3.8b` | 8 GB | Good | Low-resource systems |

### Vision Model (for diagram description)

| Model | Command | RAM | Speed (CPU) | Best For |
|-------|---------|-----|-------------|----------|
| **minicpm-v** | `ollama pull minicpm-v` | ~5 GB | ~30-45s/img | Best for documents & diagrams |
| llava:7b | `ollama pull llava:7b` | ~4.7 GB | ~30-60s/img | Good general purpose |
| llava-phi3 | `ollama pull llava-phi3` | ~3 GB | ~20-30s/img | Fastest, lighter |
| llama3.2-vision | `ollama pull llama3.2-vision` | ~7 GB | ~45-60s/img | Strong technical understanding |

**Pick based on your hardware:**
- **8 GB RAM** → `phi3:3.8b` (text) — skip vision model
- **16 GB RAM** → `llama3.3:8b` (text) + `llava-phi3` (vision)
- **32 GB RAM** → `llama3.3:8b` (text) + `minicpm-v` (vision) — recommended
- **48+ GB RAM** → `llama3.3:70b` (text) + `minicpm-v` (vision) — best quality

---

## Quick Start

### 1. System Dependencies

```bash
# Tesseract OCR (fallback for diagrams without vision model)
sudo apt-get install tesseract-ocr    # Ubuntu/Debian
brew install tesseract                  # macOS
```

### 2. Ollama (Local LLM + Vision)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve                             # Start server (keep running)
ollama pull llama3.3:8b                  # Pull text LLM
ollama pull minicpm-v                    # Pull vision model for diagrams
```

### 3. Python Setup

```bash
pip install -r requirements.txt
```

### 4. Launch

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Usage Flow

1. **Register Equipment** — `Equipment Manager` → enter ID + name → register
2. **Upload Manuals** — `Upload Manuals` → select equipment → drop PDFs → process
3. **Ask Questions** — `Diagnostic Chat` → type question → get streaming answer with citations

The pipeline is automatic: upload → extract text/tables/diagrams → detect sections → describe diagrams with vision → semantic chunking → embed → store. No extra steps.

**Vision toggle:** In `Advanced Settings` sidebar, uncheck "Use Vision Model for Diagrams" for faster OCR-only processing.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **PDF Text** | PyMuPDF | Fastest Python PDF parser |
| **Tables** | pdfplumber | Best table extraction for technical docs |
| **Diagram Description** | Ollama Vision (minicpm-v / LLaVA) | Rich technical descriptions, offline |
| **OCR Fallback** | Tesseract | Proven, offline, handles text-in-images |
| **Spatial Context** | PyMuPDF bounding boxes | Links diagrams to surrounding text |
| **Section Detection** | Custom regex | Detects CHAPTER, numbered sections, ALL CAPS headers |
| **Embeddings** | sentence-transformers (BGE-small) | Best retrieval accuracy at small size |
| **Vector DB** | ChromaDB | Embedded, persistent, equipment-isolated |
| **LLM** | Ollama (Llama 3.3 / Qwen 2.5 / etc.) | Local, no API keys, GPU optional |
| **UI** | Streamlit | Native chat components, streaming support |

---

## Project Structure

```
manual-diagnostic-ai/
├── app.py              # Streamlit UI — 4 tabs: Chat, Equipment, Upload, Guide
├── doc_processor.py    # PDF extraction + vision diagrams + section detection + semantic chunking
├── vector_store.py     # ChromaDB with equipment isolation + rich metadata + image paths
├── llm_engine.py       # Ollama integration + 9 model recommendations + diagnostic prompt
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template (LLM, vision, embedding, storage)
├── .gitignore
├── LICENSE             # MIT
├── README.md
└── images/             # Extracted diagram images (auto-created, gitignored)
    └── {equipment_id}/ # One folder per equipment
```

---

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.3:8b` | Default text LLM for chat |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for vector search |
| `VISION_MODEL` | `minicpm-v` | Vision model for diagram description |
| `IMAGE_STORE_DIR` | `./images` | Where extracted diagram images are saved |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |

---

## Hardware Requirements

| Component | Minimum | Recommended (32GB) | Best |
|-----------|---------|-------------------|------|
| **RAM** | 8 GB | 32 GB | 48+ GB |
| **CPU** | 4 cores | 8+ cores | 12+ cores |
| **Storage** | 20 GB free | 50 GB SSD | 100+ GB NVMe |
| **GPU** | Not required | NVIDIA 8GB+ VRAM | NVIDIA 16-24GB VRAM |

**32GB RAM / No GPU setup** (our target):
- Text LLM: `llama3.3:8b` (~8GB) — fast enough on CPU
- Vision model: `minicpm-v` (~5GB) — ~30-45s per diagram on CPU
- Both can run together with headroom to spare

---

## Data Privacy

- **100% offline** after setup
- **No cloud APIs** — all AI runs locally via Ollama
- **Equipment isolation** — separate ChromaDB collections
- **No telemetry** — disabled by default
- **Images stored locally** — extracted diagrams saved in `./images/`, never uploaded
- **Delete anytime** — remove equipment and all its data (vectors + images) in one click

---

## Changelog

### v2.1 (2026-02-12) — Vision-Aware Diagram Processing
- **Vision model integration** — Ollama vision models (minicpm-v/LLaVA) describe engineering diagrams during PDF ingestion
- **Surrounding text extraction** — bounding-box geometry captures text above/below/caption near diagrams
- **Image storage** — extracted diagrams saved to disk as PNG in `./images/{equipment_id}/`
- **Image deduplication** — MD5 hash skips duplicate images within a PDF
- **Graceful fallback** — vision → Tesseract OCR → skip (never crashes)
- **Vision toggle** — checkbox in Advanced Settings to disable vision for fast OCR-only processing
- **Image path metadata** — stored in ChromaDB for future inline display
- **Image cleanup** — deleting equipment also removes its saved diagram images

### v2.0 (2026-02-11) — Semantic Chunking & Streaming
- Section-aware semantic chunking with full hierarchy
- Streaming chat responses via Ollama
- 9 recommended model profiles
- Equipment-isolated ChromaDB collections
- Auto-pipeline: upload → extract → embed → store

### v1.0 (2026-02-09) — Initial Release
- Basic PDF text/table extraction
- Tesseract OCR for images
- Ollama LLM integration
- Streamlit UI

---

## License

MIT — use it, modify it, deploy it. Built for engineers who need answers from their manuals, not from the internet.
