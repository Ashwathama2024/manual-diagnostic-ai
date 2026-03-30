from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import Any

from common import ensure_ollama_models, ensure_python_package, load_config, require_path, resolve_path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_prompt(question: str, contexts: list[dict[str, Any]], retrieval_mode: str) -> str:
    context_blocks: list[str] = []
    for i, chunk in enumerate(contexts, start=1):
        section = chunk.get("section_path", "")
        text = chunk.get("text", "")
        score = chunk.get("score")
        score_line = f"Score: {score}\n" if score is not None else ""
        context_blocks.append(f"[Chunk {i}]\nSection: {section}\n{score_line}{text}")

    joined = "\n\n".join(context_blocks)

    return dedent(
        f"""
        You are a technical troubleshooting assistant.
        Use only the provided context chunks.
        Retrieval mode used: {retrieval_mode}.

        Requirements:
        1) Provide a concise diagnosis hypothesis.
        2) Provide a numbered step-by-step procedure.
        3) Cite chunk numbers used for each key step.
        4) If information is insufficient, explicitly state what is missing.
        5) If the retrieved context appears off-topic, say so plainly before answering.

        User question:
        {question}

        Context:
        {joined}
        """
    ).strip()


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    require_path(chunks_path, "Chunks JSONL")
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def retrieve_vector(
    question: str,
    db_dir: str,
    table_name: str,
    top_k: int,
    embedding_model: str,
    notebook_id: str = "",
) -> list[dict[str, Any]]:
    import lancedb
    from langchain_ollama import OllamaEmbeddings

    db = lancedb.connect(db_dir)
    table = db.open_table(table_name)

    embedder = OllamaEmbeddings(model=embedding_model)
    q_vec = embedder.embed_query(question)

    query = table.search(q_vec)
    has_nb_col = "notebook_id" in table.schema.names
    if notebook_id and has_nb_col:
        safe_nb = notebook_id.replace("'", "''")
        query = query.where(f"notebook_id = '{safe_nb}'")
    results = query.limit(top_k).to_list()
    for row in results:
        if "_distance" in row and "score" not in row:
            row["score"] = round(float(row["_distance"]), 6)
    return results


def lexical_score(question_terms: list[str], chunk: dict[str, Any]) -> float:
    text = chunk.get("text", "")
    section = chunk.get("section_path", "")
    tokens = tokenize(f"{section} {text}")
    if not tokens or not question_terms:
        return 0.0

    counts = Counter(tokens)
    unique_matches = sum(1 for term in set(question_terms) if counts[term] > 0)
    weighted_matches = sum(math.log1p(counts[term]) for term in question_terms if counts[term] > 0)
    section_boost = sum(1 for term in set(question_terms) if term in tokenize(section))
    density = unique_matches / max(len(set(question_terms)), 1)
    return weighted_matches + (section_boost * 1.5) + density


def retrieve_vectorless(
    question: str,
    chunks_path: Path,
    top_k: int,
    notebook_id: str = "",
) -> list[dict[str, Any]]:
    chunks = load_chunks(chunks_path)
    if notebook_id:
        chunks = [c for c in chunks if c.get("notebook_id", "") == notebook_id]
    question_terms = tokenize(question)

    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        score = lexical_score(question_terms, chunk)
        if score <= 0:
            continue
        enriched = dict(chunk)
        enriched["score"] = round(score, 6)
        scored.append(enriched)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def dedupe_by_id(chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id", ""))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk)
        if len(deduped) >= top_k:
            break
    return deduped


def retrieve_hybrid(
    question: str,
    db_dir: str,
    table_name: str,
    top_k: int,
    embedding_model: str,
    chunks_path: Path,
    notebook_id: str = "",
) -> list[dict[str, Any]]:
    lexical = retrieve_vectorless(question, chunks_path, top_k=top_k * 2, notebook_id=notebook_id)
    vector = retrieve_vector(question, db_dir, table_name, top_k=top_k * 2, embedding_model=embedding_model, notebook_id=notebook_id)
    merged = lexical + vector
    return dedupe_by_id(merged, top_k)


def resolve_retrieval_mode(args_mode: str | None, cfg: dict[str, Any]) -> str:
    mode = args_mode or cfg.get("retrieval", {}).get("mode", "vector")
    allowed = {"vector", "vectorless", "hybrid"}
    if mode not in allowed:
        raise ValueError(f"Unsupported retrieval mode `{mode}`. Use one of: {', '.join(sorted(allowed))}.")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve + reason over local chunks.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "vectorless", "hybrid"],
        default=None,
        help="Override retrieval mode for this query.",
    )
    args = parser.parse_args()

    ensure_python_package("ollama")

    cfg = load_config(args.config)

    db_dir = str(resolve_path(cfg["paths"]["lancedb_dir"]))
    table_name = cfg["paths"]["lancedb_table"]
    chunks_path = resolve_path(cfg["paths"]["chunks_output"])
    embedding_model = cfg["models"]["embedding"]
    reasoning_model = cfg["models"]["reasoning"]
    top_k = args.top_k or int(cfg["retrieval"]["top_k"])
    retrieval_mode = resolve_retrieval_mode(args.retrieval_mode, cfg)

    runtime = cfg.get("runtime", {})
    if runtime.get("ollama_num_thread"):
        os.environ["OLLAMA_NUM_THREAD"] = str(runtime["ollama_num_thread"])
    if runtime.get("ollama_ctx_num"):
        os.environ["OLLAMA_CONTEXT_LENGTH"] = str(runtime["ollama_ctx_num"])

    required_models = [reasoning_model]
    if retrieval_mode in {"vector", "hybrid"}:
        ensure_python_package("lancedb")
        ensure_python_package("langchain_ollama")
        required_models.append(embedding_model)

    ensure_ollama_models(required_models)

    if retrieval_mode == "vector":
        contexts = retrieve_vector(args.question, db_dir, table_name, top_k, embedding_model)
    elif retrieval_mode == "vectorless":
        contexts = retrieve_vectorless(args.question, chunks_path, top_k)
    else:
        contexts = retrieve_hybrid(args.question, db_dir, table_name, top_k, embedding_model, chunks_path)

    if not contexts:
        raise RuntimeError("No retrieval results found. Build chunks or the index first.")

    prompt = build_prompt(args.question, contexts, retrieval_mode)

    from ollama import Client

    client = Client(host="http://127.0.0.1:11434")
    response = client.chat(
        model=reasoning_model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": int(runtime.get("ollama_ctx_num", 8192)),
            "num_thread": int(runtime.get("ollama_num_thread", 8)),
        },
    )

    print("=== Answer ===")
    print(response["message"]["content"])
    print(f"\n=== Retrieved Chunks ({retrieval_mode}) ===")
    for i, chunk in enumerate(contexts, start=1):
        section = chunk.get("section_path", "")
        score = chunk.get("score")
        score_text = f" score={score}" if score is not None else ""
        print(f"[{i}]{score_text} {section}")

    print("\n=== Retrieval Payload (JSON) ===")
    print(json.dumps(contexts, ensure_ascii=True, indent=2)[:4000])


if __name__ == "__main__":
    main()
