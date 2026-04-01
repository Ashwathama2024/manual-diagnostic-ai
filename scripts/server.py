"""
ManualIQ — FastAPI backend (replaces Gradio app.py)
====================================================
Serves the custom HTML/JS dashboard + all API endpoints.

Run:
    python scripts/server.py [--config config.yaml] [--port 7860] [--show-thinking]
or:
    uvicorn scripts.server:app --host 127.0.0.1 --port 7860
"""
from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Make scripts/ importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, resolve_path
from ingest import ingest_file
from index import index_markdown
from notebook_map import (
    get_boost_table,
    get_memory,
    regenerate_memory,
    update_map,
)

# ── Import all helpers from app.py (no duplication) ────────────────────────
from app import (
    _SYSTEM_PROMPT,          # noqa: F401 (used indirectly via build_prompt)
    build_prompt,
    call_ollama,
    create_notebook,
    find_notebook_by_id,
    find_notebook_by_name,
    format_sources_display,  # noqa: F401
    get_nb_sources,
    load_notebooks,
    nb_chunks_path,
    nb_processed_dir,
    nb_raw_dir,
    notebook_names,          # noqa: F401
    rebuild_chunks_jsonl,
    retrieve,
    save_notebooks,          # noqa: F401
    strip_thinking,
)

# ---------------------------------------------------------------------------
# Global state — populated once at startup
# ---------------------------------------------------------------------------
_cfg: dict[str, Any] = {}
_raw_dir: Path = Path("data/raw")
_processed_dir: Path = Path("data/processed")
_registry_path: Path = Path("data/notebooks.json")
_chats_dir: Path = Path("data/chats")
_maps_dir: Path = Path("data/maps")
_reasoning_model: str = "deepseek-r1:8b"
_retrieval_mode: str = "hybrid"
_history_turns: int = 6
_show_thinking: bool = False
_runtime: dict[str, Any] = {}
_ollama_client: Any = None
_lock = threading.Lock()


def _balance_sources(contexts: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Guarantee at least 1 chunk per unique source in the final top_k result.
    Works on the already-ranked list (position = quality) so it is
    score-direction-agnostic.
    """
    if top_k <= 0 or not contexts:
        return contexts[:top_k]

    # Map each source to its best-ranked (lowest index) chunk
    guaranteed: list[dict[str, Any]] = []
    guaranteed_ids: set[str] = set()
    seen_sources: set[str] = set()

    for ctx in contexts:
        src = ctx.get("source", "")
        cid = str(ctx.get("id", id(ctx)))
        if src and src not in seen_sources:
            seen_sources.add(src)
            guaranteed.append(ctx)
            guaranteed_ids.add(cid)
            if len(guaranteed) >= top_k:
                break

    # Fill remaining budget with best-ranked non-guaranteed chunks
    remaining = [c for c in contexts if str(c.get("id", id(c))) not in guaranteed_ids]
    combined  = guaranteed + remaining
    return combined[:top_k]


def _chat_path(nb_id: str) -> Path:
    return _chats_dir / f"{nb_id}.jsonl"


def _timestamps_path(nb_id: str) -> Path:
    return _processed_dir / nb_id / ".ingest_timestamps.json"


def _persist_chat_turn(nb_id: str, question: str, answer: str, citations: list[dict]) -> None:
    """Append a chat turn to the per-notebook JSONL history file (thread-safe)."""
    chat_path = _chat_path(nb_id)
    chat_path.parent.mkdir(parents=True, exist_ok=True)
    turn = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "role_user": question,
        "role_assistant": answer,
        "citations": citations,
    }
    with _lock:
        with chat_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(turn, ensure_ascii=False) + "\n")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_WATCHED_EXTS = {".pdf", ".docx", ".md"}

# Common marine/industrial abbreviation expansions used to enrich retrieval queries.
# Keeps originals too so exact matches still work.
_ABBREV_MAP: dict[str, str] = {
    "me":   "main engine",
    "ae":   "auxiliary engine",
    "fo":   "fuel oil",
    "lo":   "lube oil lubricating oil",
    "fw":   "fresh water",
    "sw":   "sea water",
    "hw":   "high pressure",
    "lp":   "low pressure",
    "hp":   "high pressure",
    "mcc":  "motor control center",
    "rpm":  "revolutions per minute speed",
    "tc":   "turbocharger",
    "scav": "scavenge scavenging",
    "exh":  "exhaust",
    "cyl":  "cylinder",
    "mcr":  "maximum continuous rating",
    "ecr":  "engine control room",
    "ig":   "inert gas",
    "pb":   "push button",
    "ocr":  "oil content recorder",
}


def _expand_query(query: str) -> str:
    """Expand abbreviations and keep originals so both exact and expanded terms are embedded."""
    words = query.split()
    expanded: list[str] = []
    for w in words:
        expanded.append(w)
        key = w.lower().strip(".,?!;:")
        if key in _ABBREV_MAP:
            expanded.append(_ABBREV_MAP[key])
    return " ".join(expanded)


class _CoreKnowledgeHandler(FileSystemEventHandler):
    """Re-ingest core knowledge whenever a supported file is added or modified."""

    def __init__(self) -> None:
        self._pending = False
        self._lock = threading.Lock()

    def _debounced_ingest(self) -> None:
        time.sleep(5)  # wait for large file writes to finish
        with self._lock:
            self._pending = False
        print("[ManualIQ] Core knowledge change detected — re-ingesting...")
        try:
            preflight_path = Path(__file__).parent / "preflight_core.py"
            subprocess.run([sys.executable, str(preflight_path)], check=True)
            print("[ManualIQ] Core knowledge re-ingestion complete.")
        except Exception as exc:
            print(f"[ManualIQ] Core knowledge re-ingestion failed: {exc}")

    def _trigger(self, path: str) -> None:
        if Path(path).suffix.lower() not in _WATCHED_EXTS:
            return
        with self._lock:
            if not self._pending:
                self._pending = True
                threading.Thread(target=self._debounced_ingest, daemon=True).start()

    def on_created(self, event) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._trigger(event.src_path)

    def on_modified(self, event) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._trigger(event.src_path)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ManualIQ", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def _startup() -> None:
    global _cfg, _raw_dir, _processed_dir, _registry_path, _chats_dir, _maps_dir
    global _reasoning_model, _retrieval_mode, _history_turns, _runtime, _ollama_client

    _cfg = load_config()
    paths = _cfg["paths"]
    _raw_dir = resolve_path(paths["raw_dir"])
    _processed_dir = resolve_path(paths["processed_dir"])
    _registry_path = resolve_path(paths.get("notebooks_registry", "data/notebooks.json"))
    _chats_dir = resolve_path(paths.get("chats_dir", "data/chats"))
    _maps_dir = resolve_path(paths.get("maps_dir", "data/maps"))
    _reasoning_model = _cfg["models"]["reasoning"]
    _retrieval_mode = _cfg["retrieval"]["mode"]
    _history_turns = int(_cfg.get("chat", {}).get("history_turns", 6))
    _runtime = _cfg.get("runtime", {})

    _raw_dir.mkdir(parents=True, exist_ok=True)
    _processed_dir.mkdir(parents=True, exist_ok=True)
    _chats_dir.mkdir(parents=True, exist_ok=True)
    _maps_dir.mkdir(parents=True, exist_ok=True)

    from ollama import Client
    _ollama_client = Client(host="http://127.0.0.1:11434")
    print(f"[ManualIQ] server ready — model={_reasoning_model}  retrieval={_retrieval_mode}")

    # Launch core knowledge ingestion automatically
    def _run_core_preflight():
        try:
            print("[ManualIQ] Triggering background core knowledge sync...")
            preflight_path = Path(__file__).parent / "preflight_core.py"
            subprocess.run([sys.executable, str(preflight_path)], check=True)
            print("[ManualIQ] Core knowledge background sync complete.")
        except Exception as e:
            print(f"[ManualIQ] Core knowledge sync failed: {e}")

    threading.Thread(target=_run_core_preflight, daemon=True).start()

    # Watch core_knowledge/ for new/modified docs and re-ingest automatically
    core_dir = Path(__file__).resolve().parent.parent / "core_knowledge"
    if core_dir.exists():
        observer = Observer()
        observer.schedule(_CoreKnowledgeHandler(), str(core_dir), recursive=True)
        observer.daemon = True
        observer.start()
        print(f"[ManualIQ] Watching {core_dir} for new core knowledge files...")


# ---------------------------------------------------------------------------
# Serve the SPA
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_spa() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CreateNotebookRequest(BaseModel):
    name: str
    equipment_category: str = ""


class UpdateNotebookRequest(BaseModel):
    custom_prompt: str = ""


class ChatRequest(BaseModel):
    message: str
    nb_id: str
    history: list[dict[str, str]] = []
    selected_sources: list[str] = []


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _require_notebook(nb_id: str) -> dict[str, Any]:
    nbs = load_notebooks(_registry_path)
    nb = find_notebook_by_id(nbs, nb_id)
    if not nb:
        raise HTTPException(404, f"Notebook '{nb_id}' not found")
    return nb


def _nb_stats(nb_id: str) -> dict[str, int]:
    proc = nb_processed_dir(_processed_dir, nb_id)
    sources = get_nb_sources(proc)
    chunks_path = nb_chunks_path(_processed_dir, nb_id)
    chunk_count = 0
    if chunks_path.exists():
        try:
            with chunks_path.open(encoding="utf-8") as fh:
                chunk_count = sum(1 for line in fh if line.strip())
        except Exception:
            pass
    return {"docs": len(sources), "chunks": chunk_count}


# ---------------------------------------------------------------------------
# Notebook endpoints
# ---------------------------------------------------------------------------
@app.get("/api/notebooks")
async def list_notebooks() -> list[dict[str, Any]]:
    return load_notebooks(_registry_path)


@app.post("/api/notebooks", status_code=201)
async def create_nb(req: CreateNotebookRequest) -> dict[str, Any]:
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Notebook name cannot be empty")
    nbs = load_notebooks(_registry_path)
    if find_notebook_by_name(nbs, name):
        raise HTTPException(409, f"Notebook '{name}' already exists")
    nb = create_notebook(name, req.equipment_category, nbs, _registry_path, _raw_dir, _processed_dir)
    if nb is None:
        raise HTTPException(500, "Failed to create notebook")
    return nb


@app.get("/api/notebooks/{nb_id}/sources")
async def list_sources(nb_id: str) -> dict[str, Any]:
    _require_notebook(nb_id)
    proc = nb_processed_dir(_processed_dir, nb_id)
    sources = get_nb_sources(proc)
    ts_path = _timestamps_path(nb_id)
    timestamps: dict[str, str] = {}
    if ts_path.exists():
        try:
            timestamps = json.loads(ts_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sources": sources, "timestamps": timestamps}


@app.get("/api/notebooks/{nb_id}/stats")
async def nb_stats(nb_id: str) -> dict[str, int]:
    _require_notebook(nb_id)
    return _nb_stats(nb_id)


@app.get("/api/notebooks/{nb_id}/history")
async def get_history(nb_id: str) -> list[dict[str, Any]]:
    _require_notebook(nb_id)
    chat_path = _chat_path(nb_id)
    if not chat_path.exists():
        return []
    turns: list[dict[str, Any]] = []
    with chat_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    turns.append(json.loads(line))
                except Exception:
                    pass
    return turns[-100:]  # cap at last 100 turns


@app.delete("/api/notebooks/{nb_id}/history")
async def clear_history(nb_id: str) -> dict[str, str]:
    _require_notebook(nb_id)
    chat_path = _chat_path(nb_id)
    with _lock:
        chat_path.write_text("", encoding="utf-8")
    return {"status": "cleared"}


@app.get("/api/notebooks/{nb_id}/memory")
async def get_nb_memory(nb_id: str) -> dict[str, str]:
    _require_notebook(nb_id)
    return {"memory": get_memory(_maps_dir, nb_id)}


@app.post("/api/notebooks/{nb_id}/memory/regenerate", status_code=202)
async def regen_nb_memory(nb_id: str) -> dict[str, str]:
    _require_notebook(nb_id)
    loop = asyncio.get_running_loop()

    def _regen() -> None:
        try:
            regenerate_memory(_maps_dir, nb_id, _ollama_client, _reasoning_model)
        except Exception as exc:
            print(f"[ManualIQ] Memory regen failed for {nb_id}: {exc}")

    loop.run_in_executor(None, _regen)
    return {"status": "regenerating"}


@app.patch("/api/notebooks/{nb_id}")
async def update_notebook(nb_id: str, req: UpdateNotebookRequest) -> dict[str, Any]:
    nbs = load_notebooks(_registry_path)
    nb = find_notebook_by_id(nbs, nb_id)
    if not nb:
        raise HTTPException(404, f"Notebook '{nb_id}' not found")
    nb["custom_prompt"] = req.custom_prompt.strip()
    from app import save_notebooks as _save_nbs
    _save_nbs(nbs, _registry_path)
    return nb


# ---------------------------------------------------------------------------
# Upload endpoint  — SSE stream with live progress
# ---------------------------------------------------------------------------
@app.post("/api/notebooks/{nb_id}/upload")
async def upload_source(nb_id: str, file: UploadFile = File(...)) -> StreamingResponse:
    return StreamingResponse(
        _ingest_generator(nb_id, file),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _ingest_generator(nb_id: str, file: UploadFile) -> AsyncGenerator[str, None]:
    # 1. Validate notebook
    try:
        _require_notebook(nb_id)
    except HTTPException as exc:
        yield _sse({"type": "error", "content": str(exc.detail)})
        return

    # 2. Validate file type and size, then save
    _ALLOWED_EXTS   = {".pdf", ".docx", ".doc"}
    _MAX_UPLOAD_MB  = 200
    _MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024

    safe_name = Path(file.filename or "upload.pdf").name
    file_ext  = Path(safe_name).suffix.lower()

    if file_ext not in _ALLOWED_EXTS:
        yield _sse({"type": "error",
                    "content": f"Unsupported file type '{file_ext}'. Please upload a PDF or DOCX file."})
        return

    nb_raw = nb_raw_dir(_raw_dir, nb_id)
    nb_proc = nb_processed_dir(_processed_dir, nb_id)
    nb_raw.mkdir(parents=True, exist_ok=True)
    nb_proc.mkdir(parents=True, exist_ok=True)

    yield _sse({"type": "progress", "msg": f"Saving {safe_name}…"})
    dest = nb_raw / safe_name
    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        size_mb = len(content) / (1024 * 1024)
        yield _sse({"type": "error",
                    "content": f"File too large ({size_mb:.0f} MB). Maximum allowed size is {_MAX_UPLOAD_MB} MB."})
        return

    dest.write_bytes(content)

    # 3. Skip-if-already-indexed cache check
    md_stem = Path(safe_name).stem + ".md"
    already_indexed = False
    try:
        import lancedb as _ldb
        _db = _ldb.connect(str(resolve_path(_cfg["paths"]["lancedb_dir"])))
        _tname = _cfg["paths"]["lancedb_table"]
        if _tname in _db.list_tables().tables:
            _t = _db.open_table(_tname)
            safe_src = md_stem.replace("'", "''")
            safe_nb = nb_id.replace("'", "''")
            has_nb = "notebook_id" in _t.schema.names
            _filter = (
                f"source = '{safe_src}' AND notebook_id = '{safe_nb}'"
                if has_nb else f"source = '{safe_src}'"
            )
            already_indexed = _t.count_rows(_filter) > 0
    except Exception:
        pass

    if already_indexed:
        yield _sse({"type": "done", "source": md_stem, "chunks": 0, "cached": True})
        return

    # 4. Ingest with live progress fed through asyncio.Queue
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[str | None] = asyncio.Queue()

    def _progress(msg: str) -> None:
        loop.call_soon_threadsafe(q.put_nowait, msg)

    ingest_result: dict[str, Any] = {}

    def _do_ingest() -> None:
        try:
            md_path = ingest_file(dest, nb_proc, "docling", _cfg, _progress)
            ingest_result["md_path"] = md_path
        except Exception as exc:
            ingest_result["error"] = str(exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=_do_ingest, daemon=True).start()

    while True:
        msg = await q.get()
        if msg is None:
            break
        yield _sse({"type": "progress", "msg": msg})

    if "error" in ingest_result:
        yield _sse({"type": "error", "content": f"Ingest failed: {ingest_result['error']}"})
        return

    md_path = ingest_result["md_path"]

    # 5. Embed + index (run in thread pool; no per-chunk progress needed)
    yield _sse({"type": "progress", "msg": "Embedding and indexing chunks…"})
    try:
        chunks_override = nb_chunks_path(_processed_dir, nb_id)
        n = await loop.run_in_executor(
            None, index_markdown, md_path, _cfg, nb_id, chunks_override
        )
    except Exception as exc:
        yield _sse({"type": "error", "content": f"Indexing failed: {exc}"})
        return

    # Write ingest timestamp sidecar
    ts_path = _timestamps_path(nb_id)
    timestamps: dict[str, str] = {}
    if ts_path.exists():
        try:
            timestamps = json.loads(ts_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    timestamps[md_path.name] = datetime.datetime.utcnow().isoformat() + "Z"
    ts_path.write_text(json.dumps(timestamps, indent=2, ensure_ascii=False), encoding="utf-8")

    yield _sse({"type": "done", "source": md_path.name, "chunks": n})


# ---------------------------------------------------------------------------
# Remove source
# ---------------------------------------------------------------------------
@app.delete("/api/notebooks/{nb_id}/sources/{source_name}")
async def remove_source(nb_id: str, source_name: str) -> dict[str, Any]:
    _require_notebook(nb_id)

    import lancedb
    lancedb_dir = resolve_path(_cfg["paths"]["lancedb_dir"])
    table_name = _cfg["paths"]["lancedb_table"]

    try:
        db = lancedb.connect(str(lancedb_dir))
        if table_name in db.list_tables().tables:
            table = db.open_table(table_name)
            has_nb = "notebook_id" in table.schema.names
            safe_src = source_name.replace("'", "''")
            safe_nb = nb_id.replace("'", "''")
            if has_nb:
                table.delete(f"source = '{safe_src}' AND notebook_id = '{safe_nb}'")
            else:
                table.delete(f"source = '{safe_src}'")
    except Exception as exc:
        raise HTTPException(500, f"LanceDB error: {exc}") from exc

    nb_proc = nb_processed_dir(_processed_dir, nb_id)
    nb_raw = nb_raw_dir(_raw_dir, nb_id)

    md_path = nb_proc / source_name
    if md_path.exists():
        md_path.unlink()

    pdf_stem = source_name.replace(".md", "")
    for p in nb_raw.glob(f"{pdf_stem}.*"):
        try:
            p.unlink()
        except Exception:
            pass

    chunks_path = nb_chunks_path(_processed_dir, nb_id)
    rebuild_chunks_jsonl(nb_proc, chunks_path, _cfg, nb_id)

    return {"status": "removed", "source": source_name}


# ---------------------------------------------------------------------------
# Delete notebook
# ---------------------------------------------------------------------------
@app.delete("/api/notebooks/{nb_id}")
async def delete_notebook(nb_id: str) -> dict[str, Any]:
    nb = _require_notebook(nb_id)

    # Remove from registry
    nbs = load_notebooks(_registry_path)
    nbs = [n for n in nbs if n["id"] != nb_id]
    from app import save_notebooks as _save_nbs
    _save_nbs(nbs, _registry_path)

    # Delete LanceDB rows for this notebook
    import lancedb
    lancedb_dir = resolve_path(_cfg["paths"]["lancedb_dir"])
    table_name = _cfg["paths"]["lancedb_table"]
    try:
        db = lancedb.connect(str(lancedb_dir))
        if table_name in db.list_tables().tables:
            table = db.open_table(table_name)
            if "notebook_id" in table.schema.names:
                safe_nb = nb_id.replace("'", "''")
                table.delete(f"notebook_id = '{safe_nb}'")
    except Exception:
        pass

    # Delete raw + processed directories
    for d in [nb_raw_dir(_raw_dir, nb_id), nb_processed_dir(_processed_dir, nb_id)]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    return {"status": "deleted", "id": nb_id, "name": nb["name"]}


# ---------------------------------------------------------------------------
# ThinkStripper — state machine for in-stream <think> filtering
# ---------------------------------------------------------------------------
class ThinkStripper:
    """Filter <think>...</think> blocks from a streaming token feed."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        """Feed a token; returns characters safe to emit."""
        self._buf += token
        result: list[str] = []

        while self._buf:
            if not self._in_think:
                idx = self._buf.lower().find(self.OPEN_TAG)
                if idx == -1:
                    # No opening tag — safe to emit all but last len(OPEN_TAG)-1 chars
                    safe = max(0, len(self._buf) - (len(self.OPEN_TAG) - 1))
                    result.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break
                else:
                    result.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self.OPEN_TAG):]
                    self._in_think = True
            else:
                idx = self._buf.lower().find(self.CLOSE_TAG)
                if idx == -1:
                    self._buf = ""  # discard all (still inside <think>)
                    break
                else:
                    self._buf = self._buf[idx + len(self.CLOSE_TAG):]
                    self._in_think = False

        return "".join(result)

    def flush(self) -> str:
        """Call at end of stream — emit any remaining safe buffer."""
        if self._in_think:
            self._buf = ""
            return ""
        out = self._buf
        self._buf = ""
        return out


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------
def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Chat endpoint — SSE streaming
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_generator(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _chat_generator(req: ChatRequest) -> AsyncGenerator[str, None]:
    message = req.message.strip()
    nb_id = req.nb_id
    selected_sources = req.selected_sources

    if not message:
        yield _sse({"type": "error", "content": "Empty message"})
        return
    if not nb_id:
        yield _sse({"type": "error", "content": "No notebook selected"})
        return
    if not selected_sources:
        yield _sse({"type": "error", "content": "No sources selected. Please select at least one source."})
        return

    proc = nb_processed_dir(_processed_dir, nb_id)
    nb = _require_notebook(nb_id)
    core_category   = nb.get("equipment_category", "")
    custom_prompt   = nb.get("custom_prompt", "")
    notebook_memory = get_memory(_maps_dir, nb_id)

    # Build retrieval query enriched with recent user turns
    prior_qs = " ".join(
        m["content"] for m in (req.history[-4:] if req.history else [])
        if m.get("role") == "user"
    )
    retrieval_query = _expand_query(f"{prior_qs} {message}".strip() if prior_qs else message)

    loop = asyncio.get_running_loop()
    _top_k = int(_cfg["retrieval"]["top_k"])

    try:
        # Fetch 2× pool for load balancing, then trim to top_k after balancing
        contexts = await loop.run_in_executor(
            None, retrieve, retrieval_query, _cfg, _retrieval_mode, nb_id, proc,
            selected_sources, core_category, _top_k * 2
        )
    except Exception as exc:
        yield _sse({"type": "error", "content": f"Retrieval error: {exc}"})
        return

    if not contexts:
        yield _sse({"type": "token", "content": "No relevant content found in the selected sources."})
        yield _sse({"type": "done", "citations": []})
        return

    # Load balance: guarantee ≥1 chunk per source, then trim to top_k
    if _cfg.get("retrieval", {}).get("load_balance", True):
        contexts = _balance_sources(contexts, _top_k)

    # Apply learned relevance boost — rank-based nudge (direction-agnostic)
    boost_table = get_boost_table(_maps_dir, nb_id)
    if boost_table and len(contexts) > 1:
        for i, ctx in enumerate(contexts):
            key = ctx.get("source", "unknown") + "::" + ctx.get("section_path", "")
            boost = boost_table.get(key, 1.0)
            # Effective rank: lower = better. Divide by boost so cited chunks rank higher.
            ctx["_adj_rank"] = (i + 1) / boost
        contexts.sort(key=lambda x: x["_adj_rank"])
        for ctx in contexts:
            ctx.pop("_adj_rank", None)

    # Convert [{role,content}] history to [(q, a)] tuples for build_prompt
    hist_tuples: list[tuple[str, str]] = []
    it = iter(req.history)
    for msg in it:
        if msg.get("role") == "user":
            nxt = next(it, None)
            if nxt and nxt.get("role") == "assistant":
                hist_tuples.append((msg["content"], nxt["content"]))

    prompt = build_prompt(message, contexts, hist_tuples, selected_sources, _history_turns, custom_prompt, notebook_memory)

    citations = [
        {
            "index": i,
            "source": c.get("source", ""),
            "section": c.get("section_path", ""),
            "score": c.get("score"),
        }
        for i, c in enumerate(contexts, 1)
    ]

    # Bridge: sync Ollama stream → async SSE via queue
    # Sentinel prefixes: \x00ERR: = error, \x01 = think-block started, \x02 = thinking token
    q: asyncio.Queue[str | None] = asyncio.Queue()

    def _stream_ollama() -> None:
        try:
            stream = _ollama_client.chat(
                model=_reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                think=True,  # ask Ollama to expose reasoning via chunk.message.thinking
                options={
                    "num_ctx": int(_runtime.get("ollama_ctx_num", 8192)),
                    "num_thread": int(_runtime.get("ollama_num_thread", 8)),
                },
            )
            notified = False
            for chunk in stream:
                thinking = getattr(chunk.message, "thinking", None) or ""
                content  = chunk.message.content or ""

                if thinking:
                    if not notified:
                        loop.call_soon_threadsafe(q.put_nowait, "\x01")
                        notified = True
                    loop.call_soon_threadsafe(q.put_nowait, "\x02" + thinking)

                if content:
                    loop.call_soon_threadsafe(q.put_nowait, content)

        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, f"\x00ERR:{exc}")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    threading.Thread(target=_stream_ollama, daemon=True).start()

    full_answer = ""
    while True:
        token = await q.get()
        if token is None:
            break
        if token.startswith("\x00ERR:"):
            yield _sse({"type": "error", "content": token[5:]})
            return
        if token == "\x01":
            yield _sse({"type": "thinking"})
            continue
        if token.startswith("\x02"):
            yield _sse({"type": "thinking_token", "content": token[1:]})
            continue
        full_answer += token
        yield _sse({"type": "token", "content": token})

    yield _sse({"type": "done", "citations": citations})

    # Persist chat turn + update query map after streaming completes (non-blocking)
    if full_answer:
        def _post_stream_tasks() -> None:
            _persist_chat_turn(nb_id, message, full_answer, citations)
            total = update_map(_maps_dir, nb_id, message, citations, _retrieval_mode)
            # Regenerate memory summary every 20 queries
            if total > 0 and total % 20 == 0:
                try:
                    regenerate_memory(_maps_dir, nb_id, _ollama_client, _reasoning_model)
                except Exception as exc:
                    print(f"[ManualIQ] Memory regen failed for {nb_id}: {exc}")

        threading.Thread(target=_post_stream_tasks, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ManualIQ FastAPI server")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    global _show_thinking
    _show_thinking = args.show_thinking

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
