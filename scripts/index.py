from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common import ensure_ollama_models, ensure_python_package, load_config, require_path, resolve_path


@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    section_path: str
    headers: dict[str, str]
    notebook_id: str = ""
    chunk_type: str = "text"   # "text" | "figure" | "table"


def read_markdown(path: Path) -> str:
    return require_path(path, "Markdown file").read_text(encoding="utf-8", errors="ignore")


def split_markdown(
    markdown: str,
    min_chars: int,
    max_chars: int,
    overlap: int,
    source_name: str,
    notebook_id: str = "",
) -> list[ChunkRecord]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    base_docs = header_splitter.split_text(markdown)

    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks: list[ChunkRecord] = []
    chunk_idx = 0
    for doc in base_docs:
        text = doc.page_content.strip()
        if not text:
            continue

        if min_chars <= len(text) <= max_chars:
            parts = [text]
        else:
            parts = secondary_splitter.split_text(text)

        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue

            headers = {k: str(v) for k, v in doc.metadata.items()}
            section_path = " > ".join(
                [headers.get("h1", ""), headers.get("h2", ""), headers.get("h3", ""), headers.get("h4", "")]
            ).strip(" >")

            # Detect figure chunks (injected by vision pipeline)
            if "<!-- FIGURE" in cleaned:
                ctype = "figure"
            elif cleaned.startswith("|") or "\n|" in cleaned[:120]:
                ctype = "table"
            else:
                ctype = "text"

            chunks.append(
                ChunkRecord(
                    id=f"chunk-{chunk_idx:08d}",
                    text=cleaned,
                    source=source_name,
                    section_path=section_path,
                    headers=headers,
                    notebook_id=notebook_id,
                    chunk_type=ctype,
                )
            )
            chunk_idx += 1

    return chunks


def save_jsonl(chunks: list[ChunkRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=True) + "\n")


def merge_jsonl(
    chunks: list[ChunkRecord],
    out_path: Path,
    source_name: str,
    notebook_id: str = "",
) -> None:
    """Append-merge: replace all existing records for (source_name, notebook_id), keep others."""
    existing: list[dict[str, Any]] = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    # Keep records that belong to a different source OR different notebook
                    if rec.get("source") != source_name or rec.get("notebook_id", "") != notebook_id:
                        existing.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in existing:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=True) + "\n")


def index_to_lancedb(
    chunks: list[ChunkRecord],
    db_dir: Path,
    table_name: str,
    embedding_model: str,
    batch_size: int,
    notebook_id: str = "",
) -> None:
    import lancedb
    from langchain_ollama import OllamaEmbeddings

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    embedder = OllamaEmbeddings(model=embedding_model)

    rows: list[dict[str, Any]] = []

    # Section-prefix embedding: "[Section: X > Y]\n{text}"
    # This bakes topic context into the vector so retrieval finds
    # "fuel pump specs" even when the chunk text doesn't repeat the heading.
    embed_texts = [
        f"[{chunk.section_path}]\n{chunk.text}" if chunk.section_path else chunk.text
        for chunk in chunks
    ]

    for i in range(0, len(embed_texts), batch_size):
        batch = embed_texts[i : i + batch_size]
        vectors = embedder.embed_documents(batch)

        for j, vector in enumerate(vectors):
            chunk = chunks[i + j]
            rows.append(
                {
                    "id": chunk.id,
                    "vector": vector,
                    "text": chunk.text,          # store original (no prefix)
                    "source": chunk.source,
                    "section_path": chunk.section_path,
                    "headers": json.dumps(chunk.headers, ensure_ascii=True),
                    "notebook_id": chunk.notebook_id,
                    "chunk_type": chunk.chunk_type,
                }
            )

    # LanceDB 0.30+ returns ListTablesResponse — use .tables attribute
    existing_tables: list[str] = db.list_tables().tables
    if table_name in existing_tables:
        table = db.open_table(table_name)
        # Non-destructive migrations
        if "notebook_id" not in table.schema.names:
            table.add_columns({"notebook_id": "''"})
        if "chunk_type" not in table.schema.names:
            table.add_columns({"chunk_type": "'text'"})
        # Upsert: delete old rows for this (source, notebook_id), then add new
        sources_in_batch = {row["source"] for row in rows}
        safe_nb = notebook_id.replace("'", "''")
        for src in sources_in_batch:
            safe_src = src.replace("'", "''")
            table.delete(f"source = '{safe_src}' AND notebook_id = '{safe_nb}'")
        table.add(rows)
    else:
        db.create_table(table_name, data=rows)


def index_markdown(
    md_path: Path,
    cfg: dict[str, Any],
    notebook_id: str = "",
    chunks_path_override: Path | None = None,
) -> int:
    """
    Chunk and index a single .md file into LanceDB.
    Returns the number of chunks indexed.
    Called by app.py after ingestion completes (single-file update).
    notebook_id scopes the chunks to a specific notebook.
    chunks_path_override allows per-notebook chunk files.
    """
    ensure_python_package("lancedb")
    ensure_python_package("langchain_ollama")
    embedding_model = cfg["models"]["embedding"]
    ensure_ollama_models([embedding_model])

    min_chars = int(cfg["indexing"]["min_chunk_chars"])
    max_chars = int(cfg["indexing"]["max_chunk_chars"])
    overlap = int(cfg["indexing"]["overlap_chars"])
    batch_size = int(cfg["indexing"]["batch_size"])

    chunks_path = chunks_path_override or resolve_path(cfg["paths"]["chunks_output"])
    db_dir = resolve_path(cfg["paths"]["lancedb_dir"])
    table_name = cfg["paths"]["lancedb_table"]

    markdown = read_markdown(md_path)
    chunks = split_markdown(
        markdown, min_chars, max_chars, overlap,
        source_name=md_path.name, notebook_id=notebook_id,
    )
    if not chunks:
        return 0

    merge_jsonl(chunks, chunks_path, source_name=md_path.name, notebook_id=notebook_id)
    index_to_lancedb(chunks, db_dir, table_name, embedding_model, batch_size, notebook_id=notebook_id)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Markdown files and build a local LanceDB index.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Only write chunks.jsonl and skip embedding/LanceDB creation.",
    )
    args = parser.parse_args()

    ensure_python_package("langchain_text_splitters")

    cfg = load_config(args.config)

    chunks_output = resolve_path(cfg["paths"]["chunks_output"])
    db_dir = resolve_path(cfg["paths"]["lancedb_dir"])
    table_name = cfg["paths"]["lancedb_table"]

    min_chars = int(cfg["indexing"]["min_chunk_chars"])
    max_chars = int(cfg["indexing"]["max_chunk_chars"])
    overlap = int(cfg["indexing"]["overlap_chars"])
    batch_size = int(cfg["indexing"]["batch_size"])
    embedding_model = cfg["models"]["embedding"]

    # Resolve processed directory — support both new and legacy config keys
    paths = cfg["paths"]
    if "processed_dir" in paths:
        processed_dir = resolve_path(paths["processed_dir"])
    else:
        processed_dir = resolve_path(paths["markdown_output"]).parent

    md_files = sorted(processed_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(
            f"No .md files found in {processed_dir}. "
            "Run `python scripts/ingest.py --all` first."
        )

    all_chunks: list[ChunkRecord] = []
    for md_path in md_files:
        markdown = read_markdown(md_path)
        chunks = split_markdown(markdown, min_chars=min_chars, max_chars=max_chars,
                                overlap=overlap, source_name=md_path.name)
        print(f"  {md_path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks produced. Check whether the Markdown files keep heading structure and section content.")

    # Re-assign sequential IDs across all sources
    for i, chunk in enumerate(all_chunks):
        chunk.id = f"chunk-{i:08d}"

    save_jsonl(all_chunks, chunks_output)
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunk file: {chunks_output}")

    if args.skip_vectors:
        print("Skipped vector indexing by request.")
        return

    ensure_python_package("lancedb")
    ensure_python_package("langchain_ollama")
    ensure_ollama_models([embedding_model])

    # Full rebuild: drop and recreate table with all sources
    import lancedb
    from langchain_ollama import OllamaEmbeddings

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    embedder = OllamaEmbeddings(model=embedding_model)

    rows: list[dict] = []
    embed_texts = [
        f"[{c.section_path}]\n{c.text}" if c.section_path else c.text
        for c in all_chunks
    ]
    for i in range(0, len(embed_texts), batch_size):
        batch = embed_texts[i : i + batch_size]
        vectors = embedder.embed_documents(batch)
        for j, vector in enumerate(vectors):
            chunk = all_chunks[i + j]
            rows.append({
                "id": chunk.id,
                "vector": vector,
                "text": chunk.text,
                "source": chunk.source,
                "section_path": chunk.section_path,
                "headers": json.dumps(chunk.headers, ensure_ascii=True),
                "notebook_id": chunk.notebook_id,
                "chunk_type": chunk.chunk_type,
            })

    existing_tables: list[str] = db.list_tables().tables
    if table_name in existing_tables:
        db.drop_table(table_name)
    db.create_table(table_name, data=rows)

    print(f"LanceDB: {db_dir} / table={table_name}")


if __name__ == "__main__":
    main()
