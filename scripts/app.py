"""
ManualIQ — Offline Manual Assistant
=====================================
NotebookLM-style interface with multi-notebook support.
Each notebook is an isolated workspace for a specific equipment/system.
Data persists on disk between sessions — adding a PDF is purely additive.

Runs at http://127.0.0.1:7860 — fully offline after first model download.

Usage:
    python scripts/app.py [--config config.yaml] [--show-thinking] [--port 7860]
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Any

import gradio as gr

# Ensure scripts/ is on the path so sibling imports work
sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, resolve_path
from ingest import ingest_pdf
from index import index_markdown, read_markdown, split_markdown

# ---------------------------------------------------------------------------
# Dark theme CSS — ManualIQ dashboard
# ---------------------------------------------------------------------------
DARK_CSS = """
/* ── Global ───────────────────────────────────────────────── */
body, .gradio-container, .main {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif !important;
}

/* ── Top header row ───────────────────────────────────────── */
.header-row {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border-bottom: 1px solid #30363d;
    border-radius: 10px 10px 0 0;
    padding: 4px 0 !important;
    align-items: center !important;
}
.manualiq-logo {
    padding: 8px 18px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.manualiq-logo .logo-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #79c0ff;
    letter-spacing: 1.5px;
}
.manualiq-logo .logo-sub {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 2px;
}

/* ── Active notebook label (above chat) ───────────────────── */
.nb-header {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    color: #79c0ff;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}

/* ── Stats bar ────────────────────────────────────────────── */
.stats-bar textarea {
    background-color: #21262d !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    font-family: 'Consolas', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
    min-height: 0 !important;
    padding: 3px 10px !important;
}

/* ── Panels ───────────────────────────────────────────────── */
.source-panel {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
.source-panel label, .source-panel .label-wrap {
    color: #79c0ff !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
.chat-panel {
    background-color: #0d1117 !important;
}

/* ── Chatbot bubbles (Gradio 6 selectors) ─────────────────── */
.message {
    border-radius: 10px !important;
    font-size: 0.93rem !important;
    line-height: 1.6 !important;
}
.message.user, [data-testid="user"] .message {
    background-color: #1c2333 !important;
    color: #c9d1d9 !important;
}
.message.bot, .message.assistant,
[data-testid="bot"] .message, [data-testid="assistant"] .message {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border-left: 3px solid #388bfd !important;
}

/* ── Input boxes ──────────────────────────────────────────── */
.gradio-textbox textarea, .gradio-textbox input {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 0.93rem !important;
}
.gradio-textbox textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 2px rgba(56,139,253,0.2) !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.send-btn button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
}
.send-btn button:hover { background: #2ea043 !important; }
.clear-btn button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
.create-nb-btn button {
    background: #1f4870 !important;
    color: #79c0ff !important;
    border: 1px solid #388bfd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.create-nb-btn button:hover { background: #264f7a !important; }
.remove-btn button {
    background: #6e1a1a !important;
    color: #ffa198 !important;
    border: 1px solid #f85149 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.remove-btn button:hover { background: #8b2020 !important; }

/* ── Upload area ──────────────────────────────────────────── */
.gradio-file {
    background-color: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
}
.gradio-file:hover { border-color: #388bfd !important; }

/* ── Citations box ────────────────────────────────────────── */
.citations-box textarea {
    background-color: #161b22 !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    font-family: 'Consolas', 'Monaco', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}

/* ── Dropdown ─────────────────────────────────────────────── */
.gradio-dropdown select, .gradio-dropdown input {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Status box ───────────────────────────────────────────── */
.status-box textarea {
    background-color: #161b22 !important;
    color: #3fb950 !important;
    border: 1px solid #30363d !important;
    font-family: 'Consolas', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbars ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #388bfd; }

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    font-weight: 600;
}
"""

# ---------------------------------------------------------------------------
# Thinking-token stripping (DeepSeek-R1 specific)
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ---------------------------------------------------------------------------
# Anti-hallucination prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a strictly grounded technical assistant for industrial manuals.

ABSOLUTE RULES:
1. Every factual claim MUST cite a specific [Chunk N] inline.
2. If the answer is not in the chunks, respond ONLY with:
   "I cannot find this in the loaded documents. No answer provided."
3. Partially answerable questions: answer only the parts supported by chunks; \
mark the rest with "(not found in documents)".
4. Quote numbers, part codes, torque values, and procedures verbatim from chunks.
5. Do NOT infer, extrapolate, or synthesize beyond what is explicitly stated.
6. Do NOT use training-data knowledge — only the context chunks below.\
"""


def build_prompt(
    question: str,
    contexts: list[dict[str, Any]],
    history: list[tuple[str, str]],
    sources: list[str],
    history_turns: int,
) -> str:
    chunk_blocks: list[str] = []
    for i, chunk in enumerate(contexts, start=1):
        section = chunk.get("section_path", "")
        text = chunk.get("text", "")
        score = chunk.get("score")
        score_line = f"Score: {score}\n" if score is not None else ""
        chunk_blocks.append(
            f"[Chunk {i}]\nSource: {chunk.get('source','')}\n"
            f"Section: {section}\n{score_line}{text}"
        )

    history_lines: list[str] = []
    for i, (q, a) in enumerate(history[-history_turns:], 1):
        short_a = a[:300] + "..." if len(a) > 300 else a
        history_lines.append(f"Q{i}: {q}\nA{i}: {short_a}")

    history_block = "\n\n".join(history_lines) if history_lines else "(none)"
    source_list = ", ".join(sources) if sources else "none"
    joined_chunks = "\n\n".join(chunk_blocks)

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Sources searched: {source_list}\n\n"
        f"CONTEXT:\n{joined_chunks}\n\n"
        f"CONVERSATION HISTORY (last {history_turns} turns):\n{history_block}\n\n"
        f"QUESTION:\n{question}"
    )


# ---------------------------------------------------------------------------
# Notebook registry helpers
# ---------------------------------------------------------------------------
def load_notebooks(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_notebooks(notebooks: list[dict[str, Any]], registry_path: Path) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(notebooks, indent=2, ensure_ascii=False), encoding="utf-8")


def notebook_names(notebooks: list[dict[str, Any]]) -> list[str]:
    return [nb["name"] for nb in notebooks]


def find_notebook_by_name(notebooks: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for nb in notebooks:
        if nb["name"] == name:
            return nb
    return None


def find_notebook_by_id(notebooks: list[dict[str, Any]], nb_id: str) -> dict[str, Any] | None:
    for nb in notebooks:
        if nb["id"] == nb_id:
            return nb
    return None


def create_notebook(
    name: str,
    notebooks: list[dict[str, Any]],
    registry_path: Path,
    raw_dir: Path,
    processed_dir: Path,
) -> dict[str, Any] | None:
    """Create a new notebook. Returns None if name already exists."""
    name = name.strip()
    if not name:
        return None
    if find_notebook_by_name(notebooks, name):
        return None

    nb_id = "nb_" + secrets.token_hex(6)
    nb = {"id": nb_id, "name": name, "created": str(date.today())}
    notebooks.append(nb)
    save_notebooks(notebooks, registry_path)

    # Create per-notebook subdirectories
    (raw_dir / nb_id).mkdir(parents=True, exist_ok=True)
    (processed_dir / nb_id).mkdir(parents=True, exist_ok=True)
    return nb


# ---------------------------------------------------------------------------
# Per-notebook path helpers
# ---------------------------------------------------------------------------
def nb_raw_dir(raw_dir: Path, nb_id: str) -> Path:
    return raw_dir / nb_id


def nb_processed_dir(processed_dir: Path, nb_id: str) -> Path:
    return processed_dir / nb_id


def nb_chunks_path(processed_dir: Path, nb_id: str) -> Path:
    return processed_dir / nb_id / "chunks.jsonl"


def get_nb_sources(nb_proc_dir: Path) -> list[str]:
    """Return sorted list of .md filenames in the notebook's processed dir."""
    if not nb_proc_dir.exists():
        return []
    return sorted(f.name for f in nb_proc_dir.glob("*.md"))


def format_sources_display(sources: list[str]) -> str:
    if not sources:
        return "(no sources — upload a PDF to this notebook)"
    return "\n".join(f"● {s}" for s in sources)


def rebuild_chunks_jsonl(nb_proc_dir: Path, chunks_path: Path, cfg: dict[str, Any], nb_id: str) -> None:
    min_chars = int(cfg["indexing"]["min_chunk_chars"])
    max_chars = int(cfg["indexing"]["max_chunk_chars"])
    overlap = int(cfg["indexing"]["overlap_chars"])

    all_chunks = []
    for md_path in sorted(nb_proc_dir.glob("*.md")):
        chunks = split_markdown(
            read_markdown(md_path), min_chars, max_chars, overlap,
            source_name=md_path.name, notebook_id=nb_id,
        )
        all_chunks.extend(chunks)

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        from dataclasses import asdict
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Retrieval (notebook-scoped)
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    cfg: dict[str, Any],
    retrieval_mode: str,
    nb_id: str,
    nb_proc_dir: Path,
) -> list[dict[str, Any]]:
    from query import retrieve_hybrid, retrieve_vector, retrieve_vectorless

    db_dir = str(resolve_path(cfg["paths"]["lancedb_dir"]))
    table_name = cfg["paths"]["lancedb_table"]
    chunks_path = nb_chunks_path(nb_proc_dir.parent, nb_id)
    embedding_model = cfg["models"]["embedding"]
    top_k = int(cfg["retrieval"]["top_k"])

    if retrieval_mode == "vector":
        return retrieve_vector(question, db_dir, table_name, top_k, embedding_model, notebook_id=nb_id)
    elif retrieval_mode == "vectorless":
        return retrieve_vectorless(question, chunks_path, top_k, notebook_id=nb_id)
    else:
        return retrieve_hybrid(question, db_dir, table_name, top_k, embedding_model, chunks_path, notebook_id=nb_id)


# ---------------------------------------------------------------------------
# Ollama call  (Bug 5 fix: use attribute access, not dict access)
# ---------------------------------------------------------------------------
def call_ollama(client: Any, model: str, prompt: str, runtime: dict[str, Any]) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": int(runtime.get("ollama_ctx_num", 8192)),
            "num_thread": int(runtime.get("ollama_num_thread", 8)),
        },
    )
    # Ollama Python client >= 0.2 returns a ChatResponse Pydantic model
    try:
        return response.message.content  # type: ignore[union-attr]
    except AttributeError:
        return response["message"]["content"]  # fallback for older client


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------
def build_app(cfg: dict[str, Any], show_thinking: bool) -> gr.Blocks:
    import lancedb
    from ollama import Client

    paths = cfg["paths"]
    raw_dir = resolve_path(paths["raw_dir"])
    processed_dir = resolve_path(paths["processed_dir"])
    lancedb_dir = resolve_path(paths["lancedb_dir"])
    table_name = paths["lancedb_table"]
    registry_path = resolve_path(paths.get("notebooks_registry", "data/notebooks.json"))
    reasoning_model = cfg["models"]["reasoning"]
    retrieval_mode = cfg["retrieval"]["mode"]
    history_turns = int(cfg.get("chat", {}).get("history_turns", 6))
    runtime = cfg.get("runtime", {})

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ollama_client = Client(host="http://127.0.0.1:11434")
    _lock = threading.Lock()

    # ── Helpers (closures) ─────────────────────────────────────────────────

    def _load_nbs() -> list[dict[str, Any]]:
        return load_notebooks(registry_path)

    def _nb_names() -> list[str]:
        return notebook_names(_load_nbs())

    def _sources_for(nb_id: str) -> list[str]:
        if not nb_id:
            return []
        return get_nb_sources(nb_processed_dir(processed_dir, nb_id))

    def _stats_for(nb_id: str) -> str:
        if not nb_id:
            return ""
        sources = _sources_for(nb_id)
        chunks_path = nb_chunks_path(processed_dir, nb_id)
        chunk_count = 0
        if chunks_path.exists():
            try:
                with chunks_path.open(encoding="utf-8") as fh:
                    chunk_count = sum(1 for line in fh if line.strip())
            except Exception:
                pass
        return f"Docs: {len(sources)}  ·  Chunks: {chunk_count}"

    def _nb_header_html(nb_name: str | None) -> str:
        if not nb_name:
            return '<div class="nb-header">— No notebook selected —</div>'
        return f'<div class="nb-header">📓 {nb_name}</div>'

    # ── Event handlers ─────────────────────────────────────────────────────

    def on_create_notebook(nb_name: str):
        notebooks = _load_nbs()
        nb_name = nb_name.strip()
        if not nb_name:
            names = _nb_names()
            return (
                "⚠ Enter a notebook name first.",
                gr.update(choices=names),
                gr.update(value=None),  # nb_id_state unchanged
                "",
                "(no sources)",
                gr.update(choices=[]),
                [],    # chatbot reset
                "",    # citations reset
                _nb_header_html(None),
                "",    # stats
            )

        if find_notebook_by_name(notebooks, nb_name):
            names = _nb_names()
            return (
                f"⚠ Notebook '{nb_name}' already exists.",
                gr.update(choices=names),
                gr.update(value=None),
                "",
                "(no sources)",
                gr.update(choices=[]),
                [],
                "",
                _nb_header_html(nb_name),
                "",
            )

        nb = create_notebook(nb_name, notebooks, registry_path, raw_dir, processed_dir)
        names = _nb_names()
        return (
            f"✓ Notebook '{nb_name}' created.",
            gr.update(choices=names, value=nb_name),  # notebook dropdown
            nb["id"],                                   # nb_id_state
            "",                                         # clear new_nb_input
            format_sources_display([]),                 # sources_display
            gr.update(choices=[], value=None),          # remove_dd
            [],                                         # chatbot reset
            "",                                         # citations reset
            _nb_header_html(nb_name),                   # nb_chat_header
            _stats_for(nb["id"]),                       # stats_display
        )

    def on_select_notebook(nb_name: str):
        if not nb_name:
            return (
                "",
                format_sources_display([]),
                gr.update(choices=[], value=None),
                "No notebook selected.",
                [],
                "",
                _nb_header_html(None),
                "",
            )
        notebooks = _load_nbs()
        nb = find_notebook_by_name(notebooks, nb_name)
        if not nb:
            return (
                "",
                format_sources_display([]),
                gr.update(choices=[], value=None),
                f"⚠ Notebook '{nb_name}' not found.",
                [],
                "",
                _nb_header_html(nb_name),
                "",
            )
        nb_id = nb["id"]
        sources = _sources_for(nb_id)
        return (
            nb_id,                                                                          # nb_id_state
            format_sources_display(sources),                                                # sources_display
            gr.update(choices=sources, value=None),                                         # remove_dd
            f"📓 {nb_name} ({len(sources)} source{'s' if len(sources) != 1 else ''})",     # status_box
            [],                                                                             # chatbot reset
            "",                                                                             # citations reset
            _nb_header_html(nb_name),                                                       # nb_chat_header
            _stats_for(nb_id),                                                              # stats_display
        )

    def on_upload(files: list | None, nb_id: str):
        if not nb_id:
            return "⚠ Select or create a notebook first.", format_sources_display([]), gr.update(choices=[]), ""
        if not files:
            sources = _sources_for(nb_id)
            return "No files selected.", format_sources_display(sources), gr.update(choices=sources), _stats_for(nb_id)

        results: list[str] = []
        nb_raw = nb_raw_dir(raw_dir, nb_id)
        nb_proc = nb_processed_dir(processed_dir, nb_id)
        nb_raw.mkdir(parents=True, exist_ok=True)
        nb_proc.mkdir(parents=True, exist_ok=True)

        for f in files:
            try:
                pdf_src = Path(f.name)
                dest = nb_raw / pdf_src.name
                shutil.copy(pdf_src, dest)
                md_path = ingest_pdf(dest, nb_proc)
                n = index_markdown(
                    md_path, cfg,
                    notebook_id=nb_id,
                    chunks_path_override=nb_chunks_path(processed_dir, nb_id),
                )
                results.append(f"✓ {pdf_src.name} → {n} chunks")
            except Exception as exc:
                results.append(f"⚠ {Path(f.name).name}: {exc}")

        sources = _sources_for(nb_id)
        return "\n".join(results), format_sources_display(sources), gr.update(choices=sources, value=None), _stats_for(nb_id)

    def on_send(message: str, history: list, nb_id: str):
        """
        Bug 1 fix: history uses Gradio 6 messages format — list of dicts
                   {role: "user"|"assistant", content: str}
        Bug 2 fix: history unpacking uses dict access, not tuple unpacking
        """
        if not message.strip():
            yield history, "", ""
            return

        if not nb_id:
            yield history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "⚠ No notebook selected. Create or select a notebook first."},
            ], "", ""
            return

        sources = _sources_for(nb_id)
        if not sources:
            yield history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "⚠ No sources in this notebook. Upload a PDF first."},
            ], "", ""
            return

        nb_proc = nb_processed_dir(processed_dir, nb_id)

        # Enrich retrieval query with recent user questions (Bug 2 fix)
        prior_qs = " ".join(
            m["content"] for m in (history[-4:] if history else [])
            if m.get("role") == "user"
        )
        retrieval_query = f"{prior_qs} {message}".strip() if prior_qs else message

        try:
            contexts = retrieve(retrieval_query, cfg, retrieval_mode, nb_id, nb_proc)
        except Exception as exc:
            yield history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": f"⚠ Retrieval error: {exc}"},
            ], "", ""
            return

        if not contexts:
            yield history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "No relevant content found in the loaded documents."},
            ], "", ""
            return

        # Convert Gradio 6 dict history → (q, a) tuples for build_prompt (Bug 2 fix)
        hist_tuples: list[tuple[str, str]] = []
        it = iter(history)
        for msg in it:
            if msg.get("role") == "user":
                nxt = next(it, None)
                if nxt and nxt.get("role") == "assistant":
                    hist_tuples.append((msg["content"], nxt["content"]))

        prompt = build_prompt(message, contexts, hist_tuples, sources, history_turns)

        with _lock:
            try:
                raw = call_ollama(ollama_client, reasoning_model, prompt, runtime)
            except Exception as exc:
                yield history + [
                    {"role": "user",      "content": message},
                    {"role": "assistant", "content": f"⚠ LLM error: {exc}"},
                ], "", ""
                return

        answer = strip_thinking(raw) if not show_thinking else raw

        citation_lines = [
            f"[{i}] {c.get('source', '')}  ›  {c.get('section_path', '')}  (score={c.get('score', 'n/a')})"
            for i, c in enumerate(contexts, 1)
        ]

        # Bug 1 fix: yield dict-format messages, not tuples
        yield history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ], "\n".join(citation_lines), ""

    def on_remove(source_name: str, nb_id: str):
        if not nb_id:
            return "⚠ No notebook selected.", format_sources_display([]), gr.update(choices=[]), ""
        if not source_name:
            sources = _sources_for(nb_id)
            return "No source selected.", format_sources_display(sources), gr.update(choices=sources), _stats_for(nb_id)

        try:
            db = lancedb.connect(str(lancedb_dir))
            if table_name in db.list_tables().tables:
                table = db.open_table(table_name)
                safe_src = source_name.replace("'", "''")
                safe_nb = nb_id.replace("'", "''")
                table.delete(f"source = '{safe_src}' AND notebook_id = '{safe_nb}'")
        except Exception as exc:
            sources = _sources_for(nb_id)
            return f"⚠ LanceDB error: {exc}", format_sources_display(sources), gr.update(choices=sources), _stats_for(nb_id)

        nb_proc = nb_processed_dir(processed_dir, nb_id)
        nb_raw = nb_raw_dir(raw_dir, nb_id)

        md_path = nb_proc / source_name
        if md_path.exists():
            md_path.unlink()

        pdf_stem = source_name.replace(".md", "")
        for pdf_path in nb_raw.glob(f"{pdf_stem}.*"):
            try:
                pdf_path.unlink()
            except Exception:
                pass

        chunks_path = nb_chunks_path(processed_dir, nb_id)
        rebuild_chunks_jsonl(nb_proc, chunks_path, cfg, nb_id)

        sources = _sources_for(nb_id)
        return f"Removed: {source_name}", format_sources_display(sources), gr.update(choices=sources, value=None), _stats_for(nb_id)

    def on_clear():
        return [], "", ""

    # ── Initial state ──────────────────────────────────────────────────────
    initial_nbs = _load_nbs()
    initial_names = notebook_names(initial_nbs)
    initial_nb_id = initial_nbs[0]["id"] if initial_nbs else ""
    initial_nb_name = initial_nbs[0]["name"] if initial_nbs else None
    initial_sources = _sources_for(initial_nb_id) if initial_nb_id else []

    # ── Build UI ───────────────────────────────────────────────────────────
    with gr.Blocks(title="ManualIQ") as demo:

        nb_id_state = gr.State(initial_nb_id)

        # ── Top header row: logo + notebook selector + create ──────────────
        with gr.Row(elem_classes=["header-row"]):

            with gr.Column(scale=3, min_width=220):
                gr.HTML("""
                <div class="manualiq-logo">
                    <span class="logo-title">⚙ ManualIQ</span>
                    <span class="logo-sub">Offline reasoning over technical manuals · zero hallucination · fully local</span>
                </div>
                """)

            with gr.Column(scale=2, min_width=180):
                nb_dropdown = gr.Dropdown(
                    choices=initial_names,
                    value=initial_nb_name,
                    label="Active Notebook",
                    interactive=True,
                )

            with gr.Column(scale=3, min_width=240):
                with gr.Row():
                    new_nb_input = gr.Textbox(
                        placeholder="New notebook name...",
                        label="",
                        show_label=False,
                        lines=1,
                        scale=3,
                    )
                    create_nb_btn = gr.Button("＋ Create", scale=1, elem_classes=["create-nb-btn"])

        # ── Main content row ───────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left sidebar: Sources ──────────────────────────────────────
            with gr.Column(scale=1, min_width=260, elem_classes=["source-panel"]):

                gr.HTML('<div class="section-label">📚 Sources</div>')

                upload = gr.File(
                    label="Upload Manual(s)",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                    value="",
                    elem_classes=["status-box"],
                )
                stats_display = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=1,
                    value=_stats_for(initial_nb_id),
                    elem_classes=["stats-bar"],
                )
                sources_display = gr.Textbox(
                    label="Loaded Sources",
                    interactive=False,
                    lines=8,
                    value=format_sources_display(initial_sources),
                )
                gr.HTML('<div class="section-label" style="margin-top:12px">Remove Source</div>')
                remove_dd = gr.Dropdown(
                    choices=initial_sources,
                    value=None,
                    label="",
                    show_label=False,
                )
                remove_btn = gr.Button(
                    "🗑 Remove Selected",
                    elem_classes=["remove-btn"],
                )

            # ── Right panel: Chat ──────────────────────────────────────────
            with gr.Column(scale=3, elem_classes=["chat-panel"]):

                # Active notebook label
                nb_chat_header = gr.HTML(
                    value=_nb_header_html(initial_nb_name),
                )

                chatbot = gr.Chatbot(
                    label="",
                    height=460,
                )
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your manuals...",
                    label="",
                    show_label=False,
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("Send ➤", elem_classes=["send-btn"])
                    clear_btn = gr.Button("Clear Chat", elem_classes=["clear-btn"])

                gr.HTML('<div class="section-label" style="margin-top:10px">Retrieved Chunks (Citations)</div>')
                citations_box = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=6,
                    elem_classes=["citations-box"],
                )

        # ── Wire events ───────────────────────────────────────────────────
        # Bug 4 fix: removed duplicate status_box from on_create outputs (was at index 0 AND 6)
        # Both handlers now return 10 items matching 10 outputs
        _create_outputs = [
            status_box, nb_dropdown, nb_id_state, new_nb_input,
            sources_display, remove_dd,
            chatbot, citations_box,
            nb_chat_header, stats_display,
        ]

        create_nb_btn.click(fn=on_create_notebook, inputs=[new_nb_input], outputs=_create_outputs)
        new_nb_input.submit(fn=on_create_notebook, inputs=[new_nb_input], outputs=_create_outputs)

        nb_dropdown.change(
            fn=on_select_notebook,
            inputs=[nb_dropdown],
            outputs=[nb_id_state, sources_display, remove_dd, status_box,
                     chatbot, citations_box, nb_chat_header, stats_display],
        )

        upload.upload(
            fn=on_upload,
            inputs=[upload, nb_id_state],
            outputs=[status_box, sources_display, remove_dd, stats_display],
        )

        send_btn.click(
            fn=on_send,
            inputs=[msg_input, chatbot, nb_id_state],
            outputs=[chatbot, citations_box, msg_input],
        )
        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, chatbot, nb_id_state],
            outputs=[chatbot, citations_box, msg_input],
        )

        clear_btn.click(fn=on_clear, outputs=[chatbot, citations_box, msg_input])

        remove_btn.click(
            fn=on_remove,
            inputs=[remove_dd, nb_id_state],
            outputs=[status_box, sources_display, remove_dd, stats_display],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch ManualIQ web UI.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = load_config(args.config)
    show_thinking = args.show_thinking or cfg.get("chat", {}).get("show_thinking", False)

    demo = build_app(cfg, show_thinking)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        show_error=True,
        inbrowser=True,
        css=DARK_CSS,
    )


if __name__ == "__main__":
    main()
