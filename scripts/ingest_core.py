import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, resolve_path, ensure_python_package, ensure_ollama_models
from ingest import ingest_pdf
from index import split_markdown

log = logging.getLogger(__name__)

def process_core_knowledge(cfg, docling_exe):
    core_dir = resolve_path("core_knowledge/fundamentals")
    db_dir = resolve_path(cfg["paths"]["lancedb_dir"])
    table_name = "core_knowledge"
    
    embedding_model = cfg["models"]["embedding"]
    ensure_ollama_models([embedding_model])
    
    ensure_python_package("lancedb")
    ensure_python_package("langchain_ollama")
    import lancedb
    from langchain_ollama import OllamaEmbeddings
    
    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    embedder = OllamaEmbeddings(model=embedding_model)
    
    # 1. Gather documents
    if not core_dir.exists():
        log.warning(f"{core_dir} does not exist. Nothing to ingest.")
        return

    pdfs = list(core_dir.rglob("*.pdf"))
    docx_files = list(core_dir.rglob("*.docx"))
    docs = pdfs + docx_files
    markdowns = list(core_dir.rglob("*.md"))
    
    log.info(f"Found {len(docs)} documents ({len(pdfs)} PDFs, {len(docx_files)} DOCXs) and {len(markdowns)} existing MDs in core_knowledge.")
    
    # Ingest Docs to MD if not already done
    for doc_path in docs:
        md_path = doc_path.with_suffix(".md")
        if not md_path.exists():
            log.info(f"Ingesting {doc_path.name}...")
            ingest_pdf(doc_path, processed_dir=doc_path.parent, docling_exe=docling_exe, cfg=cfg)
            
    # Now gather all MD files
    markdowns = list(core_dir.rglob("*.md"))
    all_chunks = []
    
    min_chars = int(cfg["indexing"]["min_chunk_chars"])
    max_chars = int(cfg["indexing"]["max_chunk_chars"])
    overlap = int(cfg["indexing"]["overlap_chars"])
    batch_size = int(cfg["indexing"]["batch_size"])
    
    # 2. Chunking with metadata
    for md_path in markdowns:
        rel_path = md_path.relative_to(core_dir)
        # category is the first folder (e.g. 1_propulsion_main_machinery)
        category = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
        # core_book is the second folder (e.g. main_engine), if it exists
        core_book = rel_path.parts[1] if len(rel_path.parts) > 2 else ""
        
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_markdown(text, min_chars, max_chars, overlap, source_name=md_path.name)
        
        for c in chunks:
            all_chunks.append({
                "id": c.id,  # Overwritten below
                "text": c.text,
                "source": c.source,
                "section_path": c.section_path,
                "headers": json.dumps(c.headers, ensure_ascii=True),
                "source_type": "core",
                "equipment_category": category,
                "core_book": core_book,
                "knowledge_level": "fundamental",
                "chunk_type": c.chunk_type
            })
            
    if not all_chunks:
        log.info("No core knowledge chunks found to index.")
        return
        
    for i, c in enumerate(all_chunks):
        c["id"] = f"core-chunk-{i:08d}"
        
    log.info(f"Total core chunks to index: {len(all_chunks)}")
    
    # 3. Vector Embedding
    rows = []
    embed_texts = [
        f"[{c['section_path']}]\n{c['text']}" if c["section_path"] else c["text"]
        for c in all_chunks
    ]
    
    log.info("Embedding core chunks...")
    for i in tqdm(range(0, len(embed_texts), batch_size), desc="Embedding batches"):
        batch = embed_texts[i : i + batch_size]
        vectors = embedder.embed_documents(batch)
        for j, vector in enumerate(vectors):
            chunk_data = all_chunks[i + j]
            chunk_data["vector"] = vector
            rows.append(chunk_data)
            
    # Upsert to table
    existing_tables = db.list_tables().tables
    if table_name in existing_tables:
        db.drop_table(table_name)
    db.create_table(table_name, data=rows)
    
    log.info(f"Successfully indexed {len(rows)} core chunks to {table_name}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [core] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    from common import ensure_command
    docling_exe = ensure_command("docling", "Install with `pip install docling`.")
    
    process_core_knowledge(cfg, docling_exe)
