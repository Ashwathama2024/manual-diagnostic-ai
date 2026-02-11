"""
Manual-RAG Diagnostic Assistant — v2
======================================
AI-powered equipment diagnosis from technical manuals.
Runs 100% offline using local LLM (Ollama) + local embeddings + local vector DB.

v2 Upgrades:
  - Auto-pipeline: upload → extract → embed → store (no extra clicks)
  - Streaming chat: responses appear word-by-word
  - Semantic chunking: section-aware with full manual references
  - Better model recommendations: Llama 3.3, Qwen 2.5, DeepSeek R1, Command R
  - Native Streamlit chat components

Launch:
  streamlit run app.py
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doc_processor import process_pdf, get_processing_stats, DocumentChunk
from vector_store import VectorStore
from llm_engine import (
    check_ollama_status,
    get_available_models,
    generate_response,
    generate_response_full,
    ConversationMemory,
    RECOMMENDED_MODELS,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Manual-RAG Diagnostic Assistant",
    page_icon="wrench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main .block-container { max-width: 1200px; padding-top: 1rem; }

    .equipment-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    .stat-box {
        background: #0f3460;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        margin: 0.3rem;
    }
    .stat-box h3 { margin: 0; font-size: 1.6rem; color: #e94560; }
    .stat-box p { margin: 0; font-size: 0.8rem; color: #a0a0a0; }

    .source-ref {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 4px;
        padding: 3px 10px;
        margin: 2px;
        font-size: 0.75rem;
        color: #a0a0a0;
    }

    .status-ok { color: #00d26a; font-weight: bold; }
    .status-err { color: #e94560; font-weight: bold; }

    .how-to-step {
        background: #16213e;
        border-left: 3px solid #e94560;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    .model-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }

    .pipeline-stage {
        background: #0f3460;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "vector_store": None,
        "active_equipment": None,
        "conversation_memory": ConversationMemory(),
        "chat_history": [],
        "selected_model": DEFAULT_MODEL,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "n_results": 8,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state["vector_store"] is None:
        st.session_state["vector_store"] = VectorStore()

init_session_state()


def get_vs() -> VectorStore:
    return st.session_state["vector_store"]


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## Manual-RAG Assistant")
        st.markdown("*Offline AI diagnosis from manuals*")
        st.markdown("---")

        # --- Ollama Status ---
        st.markdown("### System Status")
        ollama_status = check_ollama_status()

        if ollama_status["running"]:
            st.markdown('<span class="status-ok">OLLAMA: ONLINE</span>', unsafe_allow_html=True)
            models = ollama_status.get("models", [])
            if models:
                # Show installed models with recommendations
                current_model = st.session_state["selected_model"]
                if current_model not in models and models:
                    current_model = models[0]

                st.session_state["selected_model"] = st.selectbox(
                    "LLM Model",
                    options=models,
                    index=models.index(current_model) if current_model in models else 0,
                    help="Bigger models = better answers but slower. See System Guide for recommendations."
                )

                # Show model info if it's a recommended model
                sel = st.session_state["selected_model"]
                # Try matching with or without version tag
                model_info = RECOMMENDED_MODELS.get(sel) or RECOMMENDED_MODELS.get(sel.split(":")[0] + ":latest")
                if model_info:
                    st.caption(f"{model_info['quality']} quality | {model_info['speed']}")
            else:
                st.warning("No models installed. Run:\n`ollama pull llama3.3:8b`")
        else:
            st.markdown('<span class="status-err">OLLAMA: OFFLINE</span>', unsafe_allow_html=True)
            st.error("Start Ollama:\n```\nollama serve\n```")

        st.markdown("---")

        # --- Equipment Selector ---
        st.markdown("### Equipment")
        vs = get_vs()
        equipment_list = vs.list_equipment()

        if equipment_list:
            equip_options = {e.equipment_id: f"{e.name} ({e.chunk_count} chunks)" for e in equipment_list}
            selected = st.selectbox(
                "Select Equipment",
                options=list(equip_options.keys()),
                format_func=lambda x: equip_options[x],
                index=list(equip_options.keys()).index(st.session_state["active_equipment"])
                if st.session_state["active_equipment"] in equip_options else 0,
            )
            if selected != st.session_state["active_equipment"]:
                st.session_state["active_equipment"] = selected
                st.session_state["chat_history"] = []
                st.session_state["conversation_memory"].clear()
                st.rerun()
        else:
            st.info("No equipment yet.\nGo to **Equipment Manager**.")

        st.markdown("---")

        # --- Settings ---
        with st.expander("Advanced Settings"):
            st.session_state["chunk_size"] = st.slider(
                "Chunk Size", 200, 2000, st.session_state["chunk_size"], 100
            )
            st.session_state["chunk_overlap"] = st.slider(
                "Chunk Overlap", 50, 500, st.session_state["chunk_overlap"], 50
            )
            st.session_state["n_results"] = st.slider(
                "Context Chunks", 1, 15, st.session_state["n_results"]
            )

render_sidebar()


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab_chat, tab_equipment, tab_upload, tab_guide = st.tabs([
    "Diagnostic Chat", "Equipment Manager", "Upload Manuals", "System Guide"
])


# ===================== TAB 1: DIAGNOSTIC CHAT =============================

with tab_chat:
    st.markdown("## Diagnostic Chat")

    active_eq = st.session_state["active_equipment"]
    vs = get_vs()

    if not active_eq:
        st.info("Select or create equipment in **Equipment Manager** first.")
    else:
        equip_info = vs.get_equipment(active_eq)
        if equip_info:
            # Stats row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><h3>{equip_info.name}</h3><p>Active Equipment</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><h3>{equip_info.chunk_count}</h3><p>Knowledge Chunks</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><h3>{equip_info.manual_count}</h3><p>Manuals Loaded</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><h3>{st.session_state["selected_model"].split(":")[0]}</h3><p>LLM Model</p></div>', unsafe_allow_html=True)

        if equip_info and equip_info.chunk_count == 0:
            st.warning("No manuals uploaded for this equipment yet. Go to **Upload Manuals** tab.")
        else:
            # --- Chat using Streamlit native chat components ---
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"], avatar="wrench" if msg["role"] == "assistant" else None):
                    st.markdown(msg["content"])
                    # Show source references for assistant messages
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander(f"Sources ({len(msg['sources'])} references)", expanded=False):
                            for src in msg["sources"]:
                                hierarchy = src.get("section_hierarchy", "") or src.get("section_title", "")
                                ref = f"**{src['source_file']}**"
                                if hierarchy:
                                    ref += f" > {hierarchy}"
                                ref += f" > Page {src['page_number']}"
                                ref += f" *[{src['chunk_type']}]*"
                                st.markdown(f"- {ref}")

            # --- Chat input ---
            user_input = st.chat_input(
                f"Ask about {equip_info.name if equip_info else 'equipment'}..."
            )

            if user_input:
                # Show user message immediately
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": user_input,
                })
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Retrieve context
                with st.spinner("Searching manuals..."):
                    results = vs.query(
                        active_eq,
                        user_input,
                        n_results=st.session_state["n_results"],
                    )

                # Build sources
                sources = [
                    {
                        "source_file": r["source_file"],
                        "page_number": r["page_number"],
                        "chunk_type": r["chunk_type"],
                        "section_title": r.get("section_title", ""),
                        "section_hierarchy": r.get("section_hierarchy", ""),
                        "chapter": r.get("chapter", ""),
                    }
                    for r in results
                ]

                # Generate streaming response
                with st.chat_message("assistant", avatar="wrench"):
                    try:
                        response_text = st.write_stream(
                            generate_response(
                                question=user_input,
                                retrieved_chunks=results,
                                model=st.session_state["selected_model"],
                                equipment_name=equip_info.name if equip_info else "",
                            )
                        )

                        # Show sources
                        if sources:
                            with st.expander(f"Sources ({len(sources)} references)", expanded=False):
                                for src in sources:
                                    hierarchy = src.get("section_hierarchy", "") or src.get("section_title", "")
                                    ref = f"**{src['source_file']}**"
                                    if hierarchy:
                                        ref += f" > {hierarchy}"
                                    ref += f" > Page {src['page_number']}"
                                    ref += f" *[{src['chunk_type']}]*"
                                    st.markdown(f"- {ref}")

                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources,
                        })

                        st.session_state["conversation_memory"].add_exchange(
                            user_input, response_text, sources
                        )

                    except Exception as e:
                        error_msg = str(e)
                        if "connect" in error_msg.lower() or "refused" in error_msg.lower():
                            st.error("Cannot connect to Ollama. Start it:\n```\nollama serve\n```")
                        elif "not found" in error_msg.lower():
                            st.error(f"Model not found. Pull it:\n```\nollama pull {st.session_state['selected_model']}\n```")
                        else:
                            st.error(f"LLM Error: {error_msg}")

                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": f"*Error: {error_msg}*",
                        })

            # Clear chat button
            if st.session_state["chat_history"]:
                if st.button("Clear Chat"):
                    st.session_state["chat_history"] = []
                    st.session_state["conversation_memory"].clear()
                    st.rerun()


# ===================== TAB 2: EQUIPMENT MANAGER ===========================

with tab_equipment:
    st.markdown("## Equipment Manager")
    st.markdown("Each equipment has its own **isolated knowledge base**. "
                "Manuals for different equipment never mix.")

    vs = get_vs()

    # --- Register ---
    st.markdown("### Register New Equipment")
    with st.form("register_equipment"):
        col1, col2 = st.columns(2)
        with col1:
            eq_id = st.text_input(
                "Equipment ID",
                placeholder="e.g., main_engine_01",
                help="Unique identifier — no spaces, use underscores"
            )
        with col2:
            eq_name = st.text_input(
                "Equipment Name",
                placeholder="e.g., MAN B&W 6S50ME-C Main Engine"
            )
        eq_desc = st.text_area(
            "Description (optional)",
            placeholder="Main propulsion engine, 2-stroke, 6-cylinder...",
            height=80,
        )
        submitted = st.form_submit_button("Register Equipment", type="primary")

        if submitted:
            if not eq_id or not eq_name:
                st.error("Equipment ID and Name are required.")
            elif eq_id in [e.equipment_id for e in vs.list_equipment()]:
                st.error(f"Equipment '{eq_id}' already exists.")
            else:
                vs.register_equipment(eq_id, eq_name, eq_desc)
                st.session_state["active_equipment"] = eq_id
                st.success(f"Registered: **{eq_name}** — now upload manuals in the **Upload Manuals** tab.")
                st.rerun()

    # --- Existing ---
    st.markdown("### Registered Equipment")
    equipment_list = vs.list_equipment()

    if not equipment_list:
        st.info("No equipment registered. Use the form above.")
    else:
        for equip in equipment_list:
            with st.container():
                st.markdown(f"""<div class="equipment-card">
                    <h4>{equip.name}</h4>
                    <p><strong>ID:</strong> {equip.equipment_id} | <strong>Manuals:</strong> {equip.manual_count} | <strong>Chunks:</strong> {equip.chunk_count}</p>
                    <p>{equip.description}</p>
                </div>""", unsafe_allow_html=True)

                if equip.manuals:
                    with st.expander("Uploaded Manuals"):
                        for m in equip.manuals:
                            st.markdown(f"- {m}")

                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"Select", key=f"sel_{equip.equipment_id}"):
                        st.session_state["active_equipment"] = equip.equipment_id
                        st.session_state["chat_history"] = []
                        st.session_state["conversation_memory"].clear()
                        st.rerun()
                with col2:
                    if st.button(f"Delete", key=f"del_{equip.equipment_id}", type="secondary"):
                        vs.delete_equipment(equip.equipment_id)
                        if st.session_state["active_equipment"] == equip.equipment_id:
                            st.session_state["active_equipment"] = None
                        st.rerun()


# ===================== TAB 3: UPLOAD MANUALS ==============================

with tab_upload:
    st.markdown("## Upload Manuals")
    st.markdown("Upload PDF manuals. The system **automatically** extracts text, tables, "
                "and diagrams, then embeds and stores them — ready to query.")

    vs = get_vs()
    equipment_list = vs.list_equipment()

    if not equipment_list:
        st.warning("Register equipment first in **Equipment Manager**.")
    else:
        target_eq = st.selectbox(
            "Upload to Equipment",
            options=[e.equipment_id for e in equipment_list],
            format_func=lambda x: next(
                (e.name for e in equipment_list if e.equipment_id == x), x
            ),
        )

        st.markdown("""
        **What gets extracted automatically:**
        - Text content (procedures, descriptions, instructions)
        - Tables (clearances, specs, tolerances, torque values)
        - Images / Diagrams (OCR text extraction)
        - Section structure (chapters, sections, subsections for precise citations)
        """)

        uploaded_files = st.file_uploader(
            "Drop PDF manuals here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF manuals"
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) ready:**")
            for f in uploaded_files:
                size_mb = f.size / 1024 / 1024
                st.markdown(f"- {f.name} ({size_mb:.1f} MB)")

            # AUTO-PROCESS on button click (one click does everything)
            if st.button("Process & Store", type="primary",
                         help="Extracts text, tables, diagrams → embeds → stores in vector DB"):

                total_chunks = 0
                total_sections = 0
                progress_bar = st.progress(0.0)
                status = st.empty()
                detail_container = st.container()

                for file_idx, uploaded_file in enumerate(uploaded_files):
                    status.markdown(f'<div class="pipeline-stage">Processing: <strong>{uploaded_file.name}</strong> ({file_idx+1}/{len(uploaded_files)})</div>', unsafe_allow_html=True)

                    # Save to temp
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        # --- AUTOMATIC PIPELINE ---
                        # Step 1: Extract + semantic chunk
                        def update_progress(stage, pct):
                            overall = (file_idx + pct) / len(uploaded_files)
                            progress_bar.progress(min(overall, 1.0))
                            status.markdown(f'<div class="pipeline-stage"><strong>{uploaded_file.name}:</strong> {stage}</div>', unsafe_allow_html=True)

                        chunks = process_pdf(
                            tmp_path,
                            target_eq,
                            chunk_size=st.session_state["chunk_size"],
                            chunk_overlap=st.session_state["chunk_overlap"],
                            progress_callback=update_progress,
                        )

                        # Step 2: Embed + store (automatic)
                        status.markdown(f'<div class="pipeline-stage"><strong>{uploaded_file.name}:</strong> Embedding & storing in vector database...</div>', unsafe_allow_html=True)
                        added = vs.add_chunks(target_eq, chunks, uploaded_file.name)
                        total_chunks += added

                        # Show detailed stats
                        stats = get_processing_stats(chunks)
                        total_sections += stats.get("sections_detected", 0)

                        with detail_container:
                            st.markdown(
                                f"**{uploaded_file.name}** — "
                                f"{stats['total_chunks']} chunks | "
                                f"{stats.get('pages_covered', 0)} pages | "
                                f"{stats.get('sections_detected', 0)} sections | "
                                f"Types: {stats.get('chunks_by_type', {})}"
                            )
                            if stats.get("sections"):
                                with st.expander(f"Detected sections ({len(stats['sections'])})"):
                                    for s in stats["sections"]:
                                        st.markdown(f"- {s}")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        logger.error(f"Processing error: {e}", exc_info=True)
                    finally:
                        os.unlink(tmp_path)

                progress_bar.progress(1.0)

                equip_name = next((e.name for e in equipment_list if e.equipment_id == target_eq), target_eq)
                status.empty()
                st.success(
                    f"Done! {len(uploaded_files)} manual(s) processed:\n"
                    f"- **{total_chunks}** knowledge chunks stored\n"
                    f"- **{total_sections}** sections detected\n"
                    f"- Equipment: **{equip_name}**\n\n"
                    f"Go to **Diagnostic Chat** to start asking questions!"
                )
                st.balloons()

        # --- Existing data ---
        st.markdown("---")
        st.markdown("### Current Knowledge Base")
        stats = vs.get_collection_stats(target_eq)
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
                st.metric("Manuals", stats.get("manual_count", 0))
            with col2:
                if stats.get("manuals"):
                    st.markdown("**Uploaded Manuals:**")
                    for m in stats["manuals"]:
                        st.markdown(f"- {m}")


# ===================== TAB 4: SYSTEM GUIDE ================================

with tab_guide:
    st.markdown("## System Guide")

    st.markdown("""
### What is Manual-RAG Diagnostic Assistant?

A **100% offline AI diagnostic system** that reads your equipment's technical manuals
and answers questions using a local Large Language Model. No internet required after
initial setup. No data leaves your machine.

**Key Features:**
- **Auto-pipeline**: Upload PDF → text + tables + diagrams extracted → embedded → stored automatically
- **Semantic chunking**: Section-aware — every chunk knows its chapter, section, subsection
- **Precise citations**: Answers reference "Manual > Chapter 3 > Section 3.2.1 > Page 45" — not just "Page 45"
- **Equipment isolation**: Each machine gets its own separate knowledge base
- **Streaming chat**: Responses appear word-by-word as the LLM generates them
- **100% offline**: Ollama LLM + local embeddings + local vector DB

---

### Architecture
    """)

    st.code("""
   Upload PDF ──> Doc Processor ──> ChromaDB ──> Diagnostic Chat
                     │                  │              │
                     ├─ Text (PyMuPDF)  │              ├─ User question
                     ├─ Tables          │              ├─ Vector search (top-k)
                     │   (pdfplumber)   │              ├─ Context + question → LLM
                     ├─ Images (OCR)    │              └─ Streaming answer
                     ├─ Section detect  │                  + source citations
                     └─ Semantic chunk  │
                         with context   │
                                        ├─ Equipment A collection
                                        ├─ Equipment B collection
                                        └─ Equipment C collection
    """, language=None)

    st.markdown("""
---

### Recommended Models (Higher = Better)
    """)

    # Model recommendation table
    st.markdown("""
| Model | Command | RAM | GPU VRAM | Quality | Best For |
|-------|---------|-----|----------|---------|----------|
| **Llama 3.3 8B** | `ollama pull llama3.3:8b` | 16 GB | 8 GB | Excellent | Best all-rounder |
| **Qwen 2.5 7B** | `ollama pull qwen2.5:7b` | 16 GB | 8 GB | Excellent | Technical reasoning |
| **DeepSeek R1 8B** | `ollama pull deepseek-r1:8b` | 16 GB | 8 GB | Excellent | Chain-of-thought diagnostics |
| **Gemma 2 9B** | `ollama pull gemma2:9b` | 16 GB | 10 GB | Excellent | Instruction following |
| **Mistral 7B** | `ollama pull mistral:7b` | 16 GB | 8 GB | Very Good | Fast & reliable |
| **Command R 35B** | `ollama pull command-r:35b` | 32 GB | 16 GB | Excellent | Built for RAG citation |
| **Llama 3.3 70B** | `ollama pull llama3.3:70b` | 48 GB | 24 GB | Best | Maximum quality |
| **Qwen 2.5 72B** | `ollama pull qwen2.5:72b` | 48 GB | 24 GB | Best | Rivals GPT-4 |
| Phi-3 3.8B | `ollama pull phi3:3.8b` | 8 GB | 4 GB | Good | Low-resource systems |

**Our recommendation:**
- **16 GB RAM / no GPU**: Start with `llama3.3:8b` or `qwen2.5:7b`
- **16 GB RAM + 8GB GPU**: Same models but ~3-5x faster inference
- **32+ GB RAM / 16GB GPU**: Try `command-r:35b` — it's purpose-built for RAG
- **48+ GB RAM / 24GB GPU**: Go with `llama3.3:70b` or `qwen2.5:72b` for best quality
    """)

    st.markdown("---")
    st.markdown("### Quick Start")

    steps = [
        ("Step 1: Install Dependencies", """
```bash
# Tesseract OCR (for diagram text extraction)
sudo apt-get install tesseract-ocr    # Ubuntu/Debian
brew install tesseract                  # macOS

# Python dependencies
pip install -r requirements.txt
```"""),
        ("Step 2: Install Ollama + Model", """
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve                             # Start server (keep running)
ollama pull llama3.3:8b                  # Pull recommended model
```"""),
        ("Step 3: Launch", """
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`"""),
        ("Step 4: Register Equipment", """
**Equipment Manager** tab → Enter ID + name → **Register Equipment**

Example: `main_engine_01` / `MAN B&W 6S50ME-C`"""),
        ("Step 5: Upload Manuals", """
**Upload Manuals** tab → Select equipment → Drop PDFs → **Process & Store**

The system automatically:
1. Extracts text, tables, diagrams
2. Detects section structure (chapters, sections, subsections)
3. Creates semantic chunks with section context
4. Embeds and stores in vector database"""),
        ("Step 6: Ask Questions", """
**Diagnostic Chat** tab → Type your question → Get streaming AI response

Every answer includes source citations with manual section + page number."""),
    ]

    for title, content in steps:
        st.markdown(f'<div class="how-to-step"><strong>{title}</strong></div>', unsafe_allow_html=True)
        st.markdown(content)

    st.markdown("""
---

### Hardware Requirements

| Component | Minimum | Recommended | Best |
|-----------|---------|-------------|------|
| **CPU** | 4 cores | 8+ cores | 12+ cores |
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **Storage** | 20 GB free | 50 GB SSD | 100+ GB NVMe |
| **GPU** | Not required | NVIDIA 8GB VRAM | NVIDIA 16-24GB VRAM |
| **OS** | Linux / macOS / Windows | Ubuntu 22.04+ | Ubuntu 22.04+ |

**GPU acceleration:**
| VRAM | What you can run | Speed boost |
|------|-----------------|-------------|
| 4 GB | Phi-3 3.8B | ~3x faster |
| 8 GB | Llama 3.3 8B, Qwen 2.5 7B | ~3-5x faster |
| 16 GB | Command R 35B | ~3x faster |
| 24 GB | Llama 3.3 70B (Q4) | ~3x faster |

---

### Data Privacy

- **100% Offline** after initial setup — no network calls during use
- **No Cloud APIs** — all AI runs locally via Ollama
- **Equipment Isolation** — separate ChromaDB collections per equipment
- **No Telemetry** — ChromaDB telemetry disabled
- **Delete Anytime** — remove equipment and all its data in one click

---

### Example Questions

**Diagnostics:**
- "Exhaust temperature on cylinder 3 is 40C above mean — what are possible causes?"
- "Troubleshoot low scavenge air pressure"
- "What causes turbocharger surging and how to fix it?"

**Information Lookup:**
- "What are the main bearing clearances?"
- "Lube oil specification for the turbocharger?"
- "List all safety interlocks for the fuel oil system"

**Procedures:**
- "Step-by-step fuel injector overhaul procedure"
- "How to calibrate the cylinder pressure sensor?"
- "Torque values for cylinder head bolts"

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot connect to Ollama" | Run `ollama serve` in a terminal |
| "Model not found" | Run `ollama pull <model-name>` |
| OCR not working | Install Tesseract: `sudo apt-get install tesseract-ocr` |
| Slow responses | Use smaller model or add GPU |
| Low quality answers | Use larger model, upload more manuals |
| Out of memory | Use phi3:3.8b or add more RAM |
    """)
