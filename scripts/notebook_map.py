"""
notebook_map.py — Per-notebook query intelligence layer
========================================================
Maintains a persistent JSON map for each notebook tracking:
  - Query log (last 500 queries)
  - Chunk usage frequency (chunk_id → {count, last_seen})
  - Section usage frequency (section_path → count)
  - Topic query count

Also provides:
  - get_boost_table()  — decayed relevance multipliers for retrieval
  - update_map()       — called after every chat turn (thread-safe)
  - regenerate_memory() — LLM-generated notebook summary
"""
from __future__ import annotations

import json
import math
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAP_LOCK = threading.Lock()
_QUERY_LOG_MAX = 500
_BOOST_CAP     = 2.0
_DECAY_DAYS    = 90.0   # e-folding half-life for boost decay

# ── Boost table cache ────────────────────────────────────────────────────────
# Avoids re-reading the query-map JSON on every query.
# Entry format:  {nb_id: (boost_table_dict, monotonic_timestamp)}
# Invalidated immediately after update_map() writes new data.
_boost_cache: dict[str, tuple[dict[str, float], float]] = {}
_BOOST_CACHE_TTL = 60.0  # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _map_path(maps_dir: Path, nb_id: str) -> Path:
    return maps_dir / f"{nb_id}_query_map.json"


def _memory_path(maps_dir: Path, nb_id: str) -> Path:
    return maps_dir / f"{nb_id}_memory.md"


def _empty_map(nb_id: str) -> dict[str, Any]:
    return {
        "nb_id": nb_id,
        "total_queries": 0,
        "last_updated": _now_iso(),
        "chunk_usage": {},
        "section_usage": {},
        "query_log": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_map(maps_dir: Path, nb_id: str) -> dict[str, Any]:
    """Load the query map for a notebook, or return an empty one."""
    path = _map_path(maps_dir, nb_id)
    if not path.exists():
        return _empty_map(nb_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_map(nb_id)


def save_map(maps_dir: Path, nb_id: str, data: dict[str, Any]) -> None:
    """Persist the query map (must be called inside _MAP_LOCK)."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    _map_path(maps_dir, nb_id).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def update_map(
    maps_dir: Path,
    nb_id: str,
    query: str,
    citations: list[dict[str, Any]],
    mode: str,
) -> int:
    """
    Update the query map after a chat turn.

    Returns total_queries (used to decide when to regenerate memory).
    Thread-safe.
    """
    with _MAP_LOCK:
        data = load_map(maps_dir, nb_id)

        now = _now_iso()
        data["last_updated"] = now
        data["total_queries"] = data.get("total_queries", 0) + 1

        # Build log entry
        chunk_ids = [c.get("source", "") + ":" + str(c.get("index", "")) for c in citations]
        log_entry: dict[str, Any] = {
            "ts": now,
            "query": query[:200],
            "mode": mode,
            "chunk_ids": chunk_ids,
        }
        data.setdefault("query_log", []).append(log_entry)
        # Trim log to last N entries
        if len(data["query_log"]) > _QUERY_LOG_MAX:
            data["query_log"] = data["query_log"][-_QUERY_LOG_MAX:]

        # Update chunk usage
        chunk_usage: dict[str, Any] = data.setdefault("chunk_usage", {})
        # Update section usage
        section_usage: dict[str, int] = data.setdefault("section_usage", {})

        for c in citations:
            # Use source+index as chunk key (chunk_id not always available via citation)
            ckey = c.get("source", "unknown") + "::" + c.get("section", "")
            if ckey not in chunk_usage:
                chunk_usage[ckey] = {"count": 0, "last_seen": now}
            chunk_usage[ckey]["count"] += 1
            chunk_usage[ckey]["last_seen"] = now

            section = c.get("section", "").strip()
            if section:
                section_usage[section] = section_usage.get(section, 0) + 1

        save_map(maps_dir, nb_id, data)
        _boost_cache.pop(nb_id, None)  # force cache refresh on next query
        return data["total_queries"]


def get_boost_table(maps_dir: Path, nb_id: str) -> dict[str, float]:
    """
    Return a dict mapping chunk key → boost multiplier.

    boost = min(BOOST_CAP, 1.0 + 0.1 * log1p(count) * decay)
    decay = exp(-days_since_last_seen / DECAY_DAYS)

    Keys are "source::section" strings matching citation format.
    Result is cached for _BOOST_CACHE_TTL seconds; invalidated by update_map().
    """
    cached = _boost_cache.get(nb_id)
    if cached is not None:
        table, ts = cached
        if (time.monotonic() - ts) < _BOOST_CACHE_TTL:
            return table

    data = load_map(maps_dir, nb_id)
    chunk_usage: dict[str, Any] = data.get("chunk_usage", {})
    if not chunk_usage:
        return {}

    now = datetime.now(timezone.utc)
    boost_table: dict[str, float] = {}

    for key, info in chunk_usage.items():
        count = info.get("count", 0)
        if count == 0:
            continue
        try:
            last_seen = datetime.fromisoformat(info["last_seen"].replace("Z", "+00:00"))
            days_old = (now - last_seen).total_seconds() / 86400.0
        except Exception:
            days_old = 0.0
        decay = math.exp(-days_old / _DECAY_DAYS)
        boost = min(_BOOST_CAP, 1.0 + 0.1 * math.log1p(count) * decay)
        boost_table[key] = boost

    _boost_cache[nb_id] = (boost_table, time.monotonic())
    return boost_table


def get_memory(maps_dir: Path, nb_id: str) -> str:
    """Return the notebook memory summary, or empty string if none exists."""
    path = _memory_path(maps_dir, nb_id)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def regenerate_memory(
    maps_dir: Path,
    nb_id: str,
    ollama_client: Any,
    model: str,
) -> str:
    """
    Generate a notebook memory summary using the LLM.
    Stores it to data/maps/{nb_id}_memory.md.
    Returns the summary string.
    """
    data = load_map(maps_dir, nb_id)
    total = data.get("total_queries", 0)
    if total == 0:
        return ""

    section_usage: dict[str, int] = data.get("section_usage", {})
    top_sections = sorted(section_usage.items(), key=lambda x: x[1], reverse=True)[:12]

    recent_queries = [e.get("query", "") for e in data.get("query_log", [])[-20:]]

    sections_text = "\n".join(f"  - {s} ({n} queries)" for s, n in top_sections) or "  (none yet)"
    queries_text  = "\n".join(f"  - {q}" for q in recent_queries if q) or "  (none yet)"

    prompt = (
        "You are summarising what a technical notebook covers based on its query history.\n\n"
        f"Total queries answered: {total}\n\n"
        f"Most-queried document sections:\n{sections_text}\n\n"
        f"Recent user questions:\n{queries_text}\n\n"
        "Write a 2-4 sentence summary of what this notebook is strong at answering. "
        "Mention the equipment or system it covers, specific technical areas with good coverage, "
        "and any apparent gaps. Be concise and specific. Do not use bullet points."
    )

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"num_ctx": 2048, "num_thread": 4},
        )
        summary = (response.message.content or "").strip()
    except Exception as exc:
        summary = f"(Memory generation failed: {exc})"

    if summary:
        maps_dir.mkdir(parents=True, exist_ok=True)
        _memory_path(maps_dir, nb_id).write_text(summary, encoding="utf-8")

    return summary
