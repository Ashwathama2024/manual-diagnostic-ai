"""
Offline Manual Assistant — Terminal REPL
=========================================
Power-user fallback for chatting with indexed manuals from the command line.
Useful for scripting, testing, and environments without a browser.

Usage:
    python scripts/chat.py [--config config.yaml] [--retrieval-mode hybrid] [--show-thinking]

Commands during chat:
    /chunks    Show retrieved chunks from the last query
    /sources   List loaded document sources
    /clear     Clear conversation history
    exit       Exit the REPL
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure scripts/ is on the path so sibling imports work
sys.path.insert(0, str(Path(__file__).parent))

from common import ensure_ollama_models, load_config, resolve_path
from query import dedupe_by_id, retrieve_hybrid, retrieve_vector, retrieve_vectorless

# ---------------------------------------------------------------------------
# Thinking-token stripping (DeepSeek-R1 specific)
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ---------------------------------------------------------------------------
# Prompt builder (anti-hallucination — same rules as app.py)
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
            f"[Chunk {i}]\nSource: {chunk.get('source', '')}\n"
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
# Retrieval helper
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    cfg: dict[str, Any],
    retrieval_mode: str,
) -> list[dict[str, Any]]:
    db_dir = str(resolve_path(cfg["paths"]["lancedb_dir"]))
    table_name = cfg["paths"]["lancedb_table"]
    chunks_path = resolve_path(cfg["paths"]["chunks_output"])
    embedding_model = cfg["models"]["embedding"]
    top_k = int(cfg["retrieval"]["top_k"])

    if retrieval_mode == "vector":
        return retrieve_vector(question, db_dir, table_name, top_k, embedding_model)
    elif retrieval_mode == "vectorless":
        return retrieve_vectorless(question, chunks_path, top_k)
    else:
        return retrieve_hybrid(question, db_dir, table_name, top_k, embedding_model, chunks_path)


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------
def get_loaded_sources(processed_dir: Path) -> list[str]:
    return sorted(f.name for f in processed_dir.glob("*.md"))


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with indexed manuals in the terminal.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "vectorless", "hybrid"],
        default=None,
        help="Override retrieval mode (default: from config).",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show DeepSeek-R1 <think> reasoning blocks in output.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    reasoning_model = cfg["models"]["reasoning"]
    embedding_model = cfg["models"]["embedding"]
    retrieval_mode = args.retrieval_mode or cfg["retrieval"].get("mode", "hybrid")
    history_turns = int(cfg.get("chat", {}).get("history_turns", 6))
    show_thinking = args.show_thinking or cfg.get("chat", {}).get("show_thinking", False)
    runtime = cfg.get("runtime", {})

    processed_dir = resolve_path(cfg["paths"].get("processed_dir",
        str(resolve_path(cfg["paths"].get("markdown_output", "data/processed")).parent)))

    # Validate models
    required_models = [reasoning_model]
    if retrieval_mode in {"vector", "hybrid"}:
        required_models.append(embedding_model)
    ensure_ollama_models(required_models)

    sources = get_loaded_sources(processed_dir)
    if sources:
        print(f"\nSources loaded: {', '.join(sources)}")
    else:
        print("\nNo sources found. Run `python scripts/ingest.py --all` and `python scripts/index.py` first.")

    print(f"Retrieval mode: {retrieval_mode}  |  Model: {reasoning_model}")
    print("Commands: /chunks  /sources  /clear  exit\n")

    from ollama import Client
    client = Client(host="http://127.0.0.1:11434")

    history: list[tuple[str, str]] = []
    last_chunks: list[dict[str, Any]] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if question.lower() == "/chunks":
            if not last_chunks:
                print("  (no chunks from last query)")
            else:
                for i, c in enumerate(last_chunks, 1):
                    score = c.get("score", "n/a")
                    src = c.get("source", "")
                    section = c.get("section_path", "")
                    print(f"  [{i}] {src}  ›  {section}  (score={score})")
            continue

        if question.lower() == "/sources":
            sources = get_loaded_sources(processed_dir)
            if sources:
                for s in sources:
                    print(f"  ● {s}")
            else:
                print("  (no sources loaded)")
            continue

        if question.lower() == "/clear":
            history.clear()
            print("  Conversation history cleared.")
            continue

        # Augment retrieval query with last 2 prior user messages
        prior_qs = " ".join(q for q, _ in history[-2:])
        retrieval_query = f"{prior_qs} {question}".strip() if prior_qs else question

        sources = get_loaded_sources(processed_dir)

        try:
            contexts = retrieve(retrieval_query, cfg, retrieval_mode)
        except Exception as exc:
            print(f"Retrieval error: {exc}")
            continue

        last_chunks = contexts

        if not contexts:
            print("Assistant: No relevant content found in the loaded documents.\n")
            continue

        prompt = build_prompt(question, contexts, history, sources, history_turns)

        try:
            response = client.chat(
                model=reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": int(runtime.get("ollama_ctx_num", 8192)),
                    "num_thread": int(runtime.get("ollama_num_thread", 8)),
                },
            )
            raw = response["message"]["content"]
        except Exception as exc:
            print(f"LLM error: {exc}")
            continue

        answer = strip_thinking(raw) if not show_thinking else raw

        print(f"\nAssistant: {answer}\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
