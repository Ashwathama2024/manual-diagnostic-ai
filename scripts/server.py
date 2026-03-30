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
import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Make scripts/ importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, resolve_path
from ingest import ingest_pdf
from index import index_markdown

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
_reasoning_model: str = "deepseek-r1:8b"
_retrieval_mode: str = "hybrid"
_history_turns: int = 6
_show_thinking: bool = False
_runtime: dict[str, Any] = {}
_ollama_client: Any = None
_lock = threading.Lock()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ManualIQ", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def _startup() -> None:
    global _cfg, _raw_dir, _processed_dir, _registry_path
    global _reasoning_model, _retrieval_mode, _history_turns, _runtime, _ollama_client

    _cfg = load_config()
    paths = _cfg["paths"]
    _raw_dir = resolve_path(paths["raw_dir"])
    _processed_dir = resolve_path(paths["processed_dir"])
    _registry_path = resolve_path(paths.get("notebooks_registry", "data/notebooks.json"))
    _reasoning_model = _cfg["models"]["reasoning"]
    _retrieval_mode = _cfg["retrieval"]["mode"]
    _history_turns = int(_cfg.get("chat", {}).get("history_turns", 6))
    _runtime = _cfg.get("runtime", {})

    _raw_dir.mkdir(parents=True, exist_ok=True)
    _processed_dir.mkdir(parents=True, exist_ok=True)

    from ollama import Client
    _ollama_client = Client(host="http://127.0.0.1:11434")
    print(f"[ManualIQ] server ready — model={_reasoning_model}  retrieval={_retrieval_mode}")


# ---------------------------------------------------------------------------
# Serve the SPA
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_spa() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CreateNotebookRequest(BaseModel):
    name: str


class ChatRequest(BaseModel):
    message: str
    nb_id: str
    history: list[dict[str, str]] = []


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
    nb = create_notebook(name, nbs, _registry_path, _raw_dir, _processed_dir)
    if nb is None:
        raise HTTPException(500, "Failed to create notebook")
    return nb


@app.get("/api/notebooks/{nb_id}/sources")
async def list_sources(nb_id: str) -> dict[str, list[str]]:
    _require_notebook(nb_id)
    proc = nb_processed_dir(_processed_dir, nb_id)
    return {"sources": get_nb_sources(proc)}


@app.get("/api/notebooks/{nb_id}/stats")
async def nb_stats(nb_id: str) -> dict[str, int]:
    _require_notebook(nb_id)
    return _nb_stats(nb_id)


# ---------------------------------------------------------------------------
# Upload endpoint  (blocking I/O offloaded to thread pool)
# ---------------------------------------------------------------------------
@app.post("/api/notebooks/{nb_id}/upload")
async def upload_source(nb_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    _require_notebook(nb_id)

    safe_name = Path(file.filename or "upload.pdf").name
    nb_raw = nb_raw_dir(_raw_dir, nb_id)
    nb_proc = nb_processed_dir(_processed_dir, nb_id)
    nb_raw.mkdir(parents=True, exist_ok=True)
    nb_proc.mkdir(parents=True, exist_ok=True)

    dest = nb_raw / safe_name
    content = await file.read()
    dest.write_bytes(content)

    # Skip-if-already-indexed: check LanceDB before running ingest+embed
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
        return {"status": "cached", "source": md_stem, "chunks": 0, "cached": True}

    loop = asyncio.get_running_loop()

    try:
        md_path = await loop.run_in_executor(None, ingest_pdf, dest, nb_proc, "docling", _cfg)
    except Exception as exc:
        raise HTTPException(500, f"Ingest failed: {exc}") from exc

    try:
        chunks_override = nb_chunks_path(_processed_dir, nb_id)
        n = await loop.run_in_executor(
            None, index_markdown, md_path, _cfg, nb_id, chunks_override
        )
    except Exception as exc:
        raise HTTPException(500, f"Indexing failed: {exc}") from exc

    return {"status": "ok", "source": md_path.name, "chunks": n}


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

    if not message:
        yield _sse({"type": "error", "content": "Empty message"})
        return
    if not nb_id:
        yield _sse({"type": "error", "content": "No notebook selected"})
        return

    proc = nb_processed_dir(_processed_dir, nb_id)
    sources = get_nb_sources(proc)
    if not sources:
        yield _sse({"type": "error", "content": "No sources in this notebook. Upload a PDF first."})
        return

    # Build retrieval query enriched with recent user turns
    prior_qs = " ".join(
        m["content"] for m in (req.history[-4:] if req.history else [])
        if m.get("role") == "user"
    )
    retrieval_query = f"{prior_qs} {message}".strip() if prior_qs else message

    loop = asyncio.get_running_loop()

    try:
        contexts = await loop.run_in_executor(
            None, retrieve, retrieval_query, _cfg, _retrieval_mode, nb_id, proc
        )
    except Exception as exc:
        yield _sse({"type": "error", "content": f"Retrieval error: {exc}"})
        return

    if not contexts:
        yield _sse({"type": "token", "content": "No relevant content found in the loaded documents."})
        yield _sse({"type": "done", "citations": []})
        return

    # Convert [{role,content}] history to [(q, a)] tuples for build_prompt
    hist_tuples: list[tuple[str, str]] = []
    it = iter(req.history)
    for msg in it:
        if msg.get("role") == "user":
            nxt = next(it, None)
            if nxt and nxt.get("role") == "assistant":
                hist_tuples.append((msg["content"], nxt["content"]))

    prompt = build_prompt(message, contexts, hist_tuples, sources, _history_turns)

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
    q: asyncio.Queue[str | None] = asyncio.Queue()
    stripper = ThinkStripper() if not _show_thinking else None

    def _stream_ollama() -> None:
        try:
            stream = _ollama_client.chat(
                model=_reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={
                    "num_ctx": int(_runtime.get("ollama_ctx_num", 8192)),
                    "num_thread": int(_runtime.get("ollama_num_thread", 8)),
                },
            )
            _notified_thinking = False
            for chunk in stream:
                raw_token = chunk.message.content or ""
                if stripper:
                    # Notify frontend once when think block starts
                    was_thinking = stripper._in_think
                    token = stripper.feed(raw_token)
                    if stripper._in_think and not was_thinking and not _notified_thinking:
                        loop.call_soon_threadsafe(q.put_nowait, "\x01THINKING")
                        _notified_thinking = True
                else:
                    token = raw_token
                if token:
                    loop.call_soon_threadsafe(q.put_nowait, token)

            # Flush remaining buffer
            if stripper:
                tail = stripper.flush()
                if tail:
                    loop.call_soon_threadsafe(q.put_nowait, tail)

        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, f"\x00ERR:{exc}")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    threading.Thread(target=_stream_ollama, daemon=True).start()

    while True:
        token = await q.get()
        if token is None:
            break
        if isinstance(token, str) and token.startswith("\x00ERR:"):
            yield _sse({"type": "error", "content": token[5:]})
            return
        if token == "\x01THINKING":
            yield _sse({"type": "thinking"})
            continue
        yield _sse({"type": "token", "content": token})

    yield _sse({"type": "done", "citations": citations})


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
