"""
Vector Store — Equipment-Isolated ChromaDB with Semantic Metadata
==================================================================
Each equipment type gets its own ChromaDB collection, ensuring
complete data isolation between different machinery.

Every chunk stores rich metadata:
  - source_file, page_number, chunk_type (text/table/image_ocr)
  - section_title, section_hierarchy, chapter
  - equipment_id

This allows the LLM to cite: "MAN B&W Manual > Chapter 3: Fuel System >
3.2.1 Injection Timing > Page 45" instead of just "Page 45".

Architecture:
  Equipment A → Collection "equip_boiler_main"
  Equipment B → Collection "equip_generator_01"
  ...each fully independent, queryable separately.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction  # Import the base class

logger = logging.getLogger(__name__)

# Where ChromaDB stores data on disk
DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

# Embedding model — BGE-small gives better retrieval accuracy than MiniLM
# while still being fast on CPU (~33M params, 384-dim)
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Embedding wrapper
# ---------------------------------------------------------------------------

class LocalEmbeddingFunction(EmbeddingFunction): # Inherit from EmbeddingFunction
    """
    Wraps sentence-transformers for ChromaDB.
    Model is downloaded once (~90MB), then runs 100% offline.

    Supports:
      - BAAI/bge-small-en-v1.5  (recommended — best retrieval at small size)
      - all-MiniLM-L6-v2         (fastest, slightly lower quality)
      - BAAI/bge-base-en-v1.5    (higher quality, slower)
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded — dimension: {self._model.get_sentence_embedding_dimension()}")

    def __call__(self, input: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(input, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def name(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# Equipment Collection Manager
# ---------------------------------------------------------------------------

@dataclass
class EquipmentInfo:
    """Metadata for a registered equipment."""
    equipment_id: str
    name: str
    description: str
    manual_count: int = 0
    chunk_count: int = 0
    manuals: list = None

    def __post_init__(self):
        if self.manuals is None:
            self.manuals = []


class VectorStore:
    """
    Equipment-isolated vector store built on ChromaDB.

    Each equipment gets its own collection. Collections are named
    with the prefix 'equip_' followed by the equipment_id.
    """

    COLLECTION_PREFIX = "equip_"
    METADATA_FILE = "equipment_registry.json"

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_fn = LocalEmbeddingFunction(embedding_model)
        self._registry = self._load_registry()

    # --- Registry management ---

    def _registry_path(self) -> str:
        return os.path.join(self.persist_dir, self.METADATA_FILE)

    def _load_registry(self) -> dict[str, dict]:
        path = self._registry_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        path = self._registry_path()
        with open(path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def _collection_name(self, equipment_id: str) -> str:
        safe_id = equipment_id.lower().replace(" ", "_").replace("-", "_")
        safe_id = "".join(c for c in safe_id if c.isalnum() or c == "_")
        name = f"{self.COLLECTION_PREFIX}{safe_id}"
        if len(name) < 3:
            name = name + "_db"
        return name[:63]

    # --- Equipment CRUD ---

    def register_equipment(self, equipment_id: str, name: str, description: str = "") -> str:
        """Register a new equipment type. Returns collection name."""
        col_name = self._collection_name(equipment_id)
        self._registry[equipment_id] = {
            "name": name,
            "description": description,
            "collection_name": col_name,
            "manual_count": 0,
            "chunk_count": 0,
            "manuals": [],
        }
        self._save_registry()
        self.client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
            metadata={"equipment_id": equipment_id, "name": name},
        )
        logger.info(f"Registered equipment '{name}' -> collection '{col_name}'")
        return col_name

    def list_equipment(self) -> list[EquipmentInfo]:
        """List all registered equipment."""
        result = []
        for eid, meta in self._registry.items():
            result.append(EquipmentInfo(
                equipment_id=eid,
                name=meta["name"],
                description=meta.get("description", ""),
                manual_count=meta.get("manual_count", 0),
                chunk_count=meta.get("chunk_count", 0),
                manuals=meta.get("manuals", []),
            ))
        return result

    def get_equipment(self, equipment_id: str) -> Optional[EquipmentInfo]:
        """Get info for a specific equipment."""
        meta = self._registry.get(equipment_id)
        if not meta:
            return None
        return EquipmentInfo(
            equipment_id=equipment_id,
            name=meta["name"],
            description=meta.get("description", ""),
            manual_count=meta.get("manual_count", 0),
            chunk_count=meta.get("chunk_count", 0),
            manuals=meta.get("manuals", []),
        )

    def delete_equipment(self, equipment_id: str) -> bool:
        """Delete an equipment and all its data (collection + saved images)."""
        meta = self._registry.get(equipment_id)
        if not meta:
            return False
        col_name = meta["collection_name"]
        try:
            self.client.delete_collection(col_name)
        except Exception as e:
            logger.warning(f"Could not delete collection {col_name}: {e}")

        # Clean up saved diagram images
        image_dir = os.path.join(
            os.environ.get("IMAGE_STORE_DIR", "./images"),
            equipment_id,
        )
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
            logger.info(f"Deleted image directory: {image_dir}")

        del self._registry[equipment_id]
        self._save_registry()
        logger.info(f"Deleted equipment '{equipment_id}'")
        return True

    # --- Document ingestion ---

    def add_chunks(self, equipment_id: str, chunks: list, source_filename: str = "") -> int:
        """
        Add document chunks to an equipment's collection.
        Stores rich metadata including section hierarchy for precise citations.

        Returns: Number of chunks added
        """
        meta = self._registry.get(equipment_id)
        if not meta:
            raise ValueError(f"Equipment '{equipment_id}' not registered. Register it first.")

        col_name = meta["collection_name"]
        collection = self.client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
        )

        # Prepare batch with rich metadata
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if not chunk.text.strip():
                continue
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_type": chunk.chunk_type,
                "equipment_id": chunk.equipment_id,
                "section_title": getattr(chunk, 'section_title', ''),
                "section_hierarchy": getattr(chunk, 'section_hierarchy', ''),
                "chapter": getattr(chunk, 'chapter', ''),
                "image_path": getattr(chunk, 'image_path', ''),
            })

        if not documents:
            return 0

        # Add in batches
        batch_size = 500
        added = 0
        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            added += len(batch_ids)

        # Update registry
        meta["chunk_count"] = collection.count()
        meta["manual_count"] = meta.get("manual_count", 0) + (1 if source_filename else 0)
        if source_filename and source_filename not in meta.get("manuals", []):
            meta.setdefault("manuals", []).append(source_filename)
        self._save_registry()

        logger.info(f"Added {added} chunks to '{equipment_id}' (total: {meta['chunk_count']})")
        return added

    # --- Querying ---

    def query(
        self,
        equipment_id: str,
        query_text: str,
        n_results: int = 8,
        chunk_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Query an equipment's knowledge base.
        Returns chunks with full section metadata for precise citations.

        Returns:
            List of {text, source_file, page_number, chunk_type,
                     section_title, section_hierarchy, chapter, distance}
        """
        meta = self._registry.get(equipment_id)
        if not meta:
            raise ValueError(f"Equipment '{equipment_id}' not registered.")

        col_name = meta["collection_name"]
        collection = self.client.get_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
        )

        if collection.count() == 0:
            return []

        where_filter = None
        if chunk_types:
            if len(chunk_types) == 1:
                where_filter = {"chunk_type": chunk_types[0]}
            else:
                where_filter = {"chunk_type": {"$in": chunk_types}}

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count()),
            where=where_filter,
        )

        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta_item = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                formatted.append({
                    "text": doc,
                    "source_file": meta_item.get("source_file", ""),
                    "page_number": meta_item.get("page_number", 0),
                    "chunk_type": meta_item.get("chunk_type", ""),
                    "section_title": meta_item.get("section_title", ""),
                    "section_hierarchy": meta_item.get("section_hierarchy", ""),
                    "chapter": meta_item.get("chapter", ""),
                    "image_path": meta_item.get("image_path", ""),
                    "distance": round(distance, 4),
                })

        return formatted

    def get_collection_stats(self, equipment_id: str) -> dict:
        """Get detailed stats for an equipment's collection."""
        meta = self._registry.get(equipment_id)
        if not meta:
            return {}

        col_name = meta["collection_name"]
        try:
            collection = self.client.get_collection(
                name=col_name,
                embedding_function=self.embedding_fn,
            )
            return {
                "equipment_id": equipment_id,
                "name": meta["name"],
                "collection_name": col_name,
                "total_chunks": collection.count(),
                "manual_count": meta.get("manual_count", 0),
                "manuals": meta.get("manuals", []),
            }
        except Exception:
            return {"equipment_id": equipment_id, "error": "Collection not found"}

    def reset_all(self):
        """Delete all data. Use with caution."""
        for eid in list(self._registry.keys()):
            self.delete_equipment(eid)
        self._registry = {}
        self._save_registry()
        logger.info("All equipment data deleted")
