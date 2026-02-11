# Manual-RAG Diagnostic Assistant

**AI-powered equipment diagnosis from your technical manuals — runs 100% offline.**

Upload PDF manuals (text, tables, diagrams). Ask diagnostic questions. Get engineering-grade answers with precise section + page citations. No internet required. No data leaves your machine.

---

## What Makes This Different

- **Auto-pipeline** — Upload PDF → text + tables + diagrams extracted → embedded → stored. One click.
- **Semantic chunking** — Every chunk knows its chapter, section, and subsection. Not just "Page 45" but "MAN B&W Manual > Chapter 3: Fuel System > 3.2.1 Injection Timing > Page 45"
- **Equipment isolation** — Each machine gets its own vector database. Main engine data never mixes with generator data.
- **Streaming chat** — Responses appear word-by-word as the LLM generates them
- **9 recommended models** — from Phi-3 (8GB RAM) to Qwen 2.5 72B (rivals GPT-4)
- **100% offline** after setup — no cloud APIs, no telemetry, no data leaks

---

## Architecture

```
Upload PDF ──> Doc Processor ──> ChromaDB ──> Diagnostic Chat
                 │                  │              │
                 ├─ Text (PyMuPDF)  │              ├─ User question
                 ├─ Tables          │              ├─ Vector search (top-k)
                 │   (pdfplumber)   │              ├─ Context + question → LLM
                 ├─ Images (OCR)    │              └─ Streaming answer
                 ├─ Section detect  │                  + section citations
                 └─ Semantic chunk  │
                     with hierarchy │
                                    ├─ Equipment A collection
                                    ├─ Equipment B collection
                                    └─ Equipment C collection
```

**Key design decisions:**
- **Section-aware chunking** — regex-based heading detection (CHAPTER, numbered sections, ALL CAPS headers) builds a breadcrumb hierarchy for every chunk
- **Chunk prefixing** — each chunk is prefixed with its section path so the embedding captures topic context, not just raw text
- **Equipment isolation** — separate ChromaDB collections per equipment type
- **BGE embeddings** — `BAAI/bge-small-en-v1.5` gives better retrieval accuracy than MiniLM with similar speed
- **Streaming responses** — uses Streamlit's `st.write_stream` with Ollama's streaming API

---

## Recommended Models

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

**Pick based on your hardware:**
- **8 GB RAM** → `phi3:3.8b`
- **16 GB RAM** → `llama3.3:8b` or `qwen2.5:7b` (recommended)
- **32 GB RAM** → `command-r:35b` (purpose-built for RAG)
- **48+ GB RAM** → `llama3.3:70b` or `qwen2.5:72b` (best quality)

---

## Quick Start

### 1. System Dependencies

```bash
# Tesseract OCR
sudo apt-get install tesseract-ocr    # Ubuntu/Debian
brew install tesseract                  # macOS
```

### 2. Ollama (Local LLM)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve                             # Start server (keep running)
ollama pull llama3.3:8b                  # Pull recommended model
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

The pipeline is automatic: upload → extract text/tables/diagrams → detect sections → semantic chunking → embed → store. No extra steps.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **PDF Text** | PyMuPDF | Fastest Python PDF parser |
| **Tables** | pdfplumber | Best table extraction for technical docs |
| **OCR** | Tesseract | Proven, offline, handles diagram text |
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
├── doc_processor.py    # PDF extraction + section detection + semantic chunking
├── vector_store.py     # ChromaDB with equipment isolation + rich metadata
├── llm_engine.py       # Ollama integration + 9 model recommendations + diagnostic prompt
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
├── .gitignore
├── LICENSE             # MIT
└── README.md
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 20 GB free | 50 GB SSD |
| **GPU** | Not required | NVIDIA 8GB+ VRAM (3-5x speed) |

---

## Data Privacy

- **100% offline** after setup
- **No cloud APIs** — all AI runs locally via Ollama
- **Equipment isolation** — separate ChromaDB collections
- **No telemetry** — disabled by default
- **Delete anytime** — remove equipment and all its data in one click

---

## License

MIT — use it, modify it, deploy it. Built for engineers who need answers from their manuals, not from the internet.
