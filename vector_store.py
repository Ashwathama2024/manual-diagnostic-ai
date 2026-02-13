"""
Vector Store — Equipment-Isolated ChromaDB with Semantic Metadata
==================================================================
Each equipment type gets its own ChromaDB collection, ensuring
complete data isolation between different machinery.

Every chunk stores rich metadata:
  - source_file, page_number, chunk_type (text/table/image_ocr)
  - section_title, section_hierarchy, chapter
  - equipment_id

Embedding Model Safety:
  - Each equipment records which embedding model + dimension was used at creation.
  - On every query, the current model is checked against the stored model.
  - If the model has changed, queries are BLOCKED with a clear warning.
  - Re-embedding is supported via re_embed_equipment() to migrate safely.

Architecture:
  Equipment A → Collection "equip_boiler_main"  (bge-small-en-v1.5, 384d)
  Equipment B → Collection "equip_generator_01"  (bge-small-en-v1.5, 384d)
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
# Retrieval Quality Configuration
# ---------------------------------------------------------------------------
# Minimum relevance score (0-100) to include a chunk in results.
# ChromaDB returns L2 distance; with normalized embeddings:
#   relevance = (1 - distance/2) * 100  →  100% = identical, 0% = unrelated
#
# Recommended thresholds:
#   40+ = reasonably relevant (default — catches most useful chunks)
#   50+ = clearly relevant (stricter — fewer but higher quality)
#   30+ = loose (more recall, risk of noise)
#   0   = disabled (return everything ChromaDB returns)
DEFAULT_MIN_RELEVANCE = float(os.environ.get("MIN_RELEVANCE_SCORE", "40"))


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class EmbeddingModelMismatchError(Exception):
    """
    Raised when the current embedding model does not match the model
    used to create an equipment's vector store.

    Querying with a different model produces semantically corrupt results.
    """
    def __init__(self, equipment_id: str, stored_model: str, stored_dim: int,
                 current_model: str, current_dim: int):
        self.equipment_id = equipment_id
        self.stored_model = stored_model
        self.stored_dim = stored_dim
        self.current_model = current_model
        self.current_dim = current_dim
        super().__init__(
            f"Embedding model mismatch for equipment '{equipment_id}': "
            f"stored={stored_model} ({stored_dim}d), "
            f"current={current_model} ({current_dim}d). "
            f"Re-embed this equipment or revert to the original model."
        )


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
            "embedding_model": self.embedding_fn.model_name,
            "embedding_dimension": self.embedding_fn.dimension,
        }
        self._save_registry()
        self.client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
            metadata={
                "equipment_id": equipment_id,
                "name": name,
                "embedding_model": self.embedding_fn.model_name,
                "embedding_dimension": self.embedding_fn.dimension,
            },
        )
        logger.info(
            f"Registered equipment '{name}' -> collection '{col_name}' "
            f"[embedding: {self.embedding_fn.model_name}, {self.embedding_fn.dimension}d]"
        )
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

    # --- Embedding model safety ---

    def _check_embedding_compatibility(self, equipment_id: str) -> None:
        """
        Verify the current embedding model matches what was used to embed
        this equipment's data. Raises EmbeddingModelMismatchError if not.

        This prevents silent semantic corruption when someone changes
        EMBEDDING_MODEL in .env without re-embedding existing data.
        """
        meta = self._registry.get(equipment_id)
        if not meta:
            return

        stored_model = meta.get("embedding_model", "")
        stored_dim = meta.get("embedding_dimension", 0)

        # Legacy equipment registered before model tracking — backfill it
        if not stored_model:
            logger.warning(
                f"Equipment '{equipment_id}' has no embedding model recorded. "
                f"Assuming current model '{self.embedding_fn.model_name}' and backfilling."
            )
            meta["embedding_model"] = self.embedding_fn.model_name
            meta["embedding_dimension"] = self.embedding_fn.dimension
            self._save_registry()
            return

        current_model = self.embedding_fn.model_name
        current_dim = self.embedding_fn.dimension

        if stored_model != current_model:
            raise EmbeddingModelMismatchError(
                equipment_id=equipment_id,
                stored_model=stored_model,
                stored_dim=stored_dim,
                current_model=current_model,
                current_dim=current_dim,
            )

        if stored_dim != current_dim:
            raise EmbeddingModelMismatchError(
                equipment_id=equipment_id,
                stored_model=stored_model,
                stored_dim=stored_dim,
                current_model=current_model,
                current_dim=current_dim,
            )

    def get_embedding_info(self, equipment_id: str) -> dict:
        """Get embedding model info for an equipment. Useful for UI display."""
        meta = self._registry.get(equipment_id)
        if not meta:
            return {}
        return {
            "model": meta.get("embedding_model", "unknown"),
            "dimension": meta.get("embedding_dimension", 0),
            "current_model": self.embedding_fn.model_name,
            "current_dimension": self.embedding_fn.dimension,
            "compatible": meta.get("embedding_model", "") == self.embedding_fn.model_name,
        }

    def re_embed_equipment(self, equipment_id: str, progress_callback=None) -> int:
        """
        Re-embed all chunks for an equipment using the CURRENT embedding model.

        Use this when you've changed the embedding model and need to migrate
        existing data. The old collection is deleted and rebuilt from stored
        document texts.

        Returns: number of chunks re-embedded
        """
        meta = self._registry.get(equipment_id)
        if not meta:
            raise ValueError(f"Equipment '{equipment_id}' not registered.")

        col_name = meta["collection_name"]

        # Step 1: Read all existing data from the old collection
        try:
            old_collection = self.client.get_collection(name=col_name)
            existing = old_collection.get(include=["documents", "metadatas"])
        except Exception as e:
            raise ValueError(f"Cannot read collection for '{equipment_id}': {e}")

        if not existing or not existing["ids"]:
            logger.warning(f"Equipment '{equipment_id}' has no chunks to re-embed.")
            return 0

        total = len(existing["ids"])
        logger.info(
            f"Re-embedding {total} chunks for '{equipment_id}': "
            f"{meta.get('embedding_model', '?')} -> {self.embedding_fn.model_name}"
        )

        # Step 2: Delete old collection
        self.client.delete_collection(col_name)

        # Step 3: Create new collection with current embedding model
        new_collection = self.client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
            metadata={
                "equipment_id": equipment_id,
                "name": meta["name"],
                "embedding_model": self.embedding_fn.model_name,
                "embedding_dimension": self.embedding_fn.dimension,
            },
        )

        # Step 4: Re-insert in batches (embeddings auto-generated by new model)
        batch_size = 100
        re_embedded = 0
        for i in range(0, total, batch_size):
            batch_ids = existing["ids"][i:i + batch_size]
            batch_docs = existing["documents"][i:i + batch_size]
            batch_meta = existing["metadatas"][i:i + batch_size] if existing["metadatas"] else [{}] * len(batch_ids)

            new_collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            re_embedded += len(batch_ids)

            if progress_callback:
                progress_callback(re_embedded / total)

        # Step 5: Update registry with new model info
        meta["embedding_model"] = self.embedding_fn.model_name
        meta["embedding_dimension"] = self.embedding_fn.dimension
        meta["chunk_count"] = new_collection.count()
        self._save_registry()

        logger.info(
            f"Re-embedded {re_embedded} chunks for '{equipment_id}' "
            f"with {self.embedding_fn.model_name} ({self.embedding_fn.dimension}d)"
        )
        return re_embedded

    # --- Document ingestion ---

    def add_chunks(self, equipment_id: str, chunks: list, source_filename: str = "") -> int:
        """
        Add document chunks to an equipment's collection.
        Stores rich metadata including section hierarchy for precise citations.

        Raises EmbeddingModelMismatchError if the current model doesn't match.
        Returns: Number of chunks added
        """
        meta = self._registry.get(equipment_id)
        if not meta:
            raise ValueError(f"Equipment '{equipment_id}' not registered. Register it first.")

        # Guard: don't mix embeddings from different models
        self._check_embedding_compatibility(equipment_id)

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

    @staticmethod
    def _l2_to_relevance(distance: float) -> float:
        """
        Convert ChromaDB L2 distance to a 0-100 relevance score.

        With normalized embeddings (normalize_embeddings=True in encode()),
        L2 distance ranges from 0 (identical) to 2 (opposite).
        Relation: cosine_similarity = 1 - (L2_distance / 2)

        Returns: relevance score 0-100 (100 = identical, 0 = unrelated)
        """
        return max(0.0, min(100.0, (1.0 - distance / 2.0) * 100.0))

    def query(
        self,
        equipment_id: str,
        query_text: str,
        n_results: int = 8,
        chunk_types: Optional[list[str]] = None,
        min_relevance: float = None,
    ) -> list[dict]:
        """
        Query an equipment's knowledge base with relevance filtering.

        Args:
            equipment_id: Which equipment to search
            query_text: The user's question
            n_results: Max chunks to retrieve from ChromaDB (before filtering)
            chunk_types: Optional filter for chunk types
            min_relevance: Minimum relevance score (0-100) to keep a chunk.
                          Chunks below this threshold are discarded as noise.
                          Default: DEFAULT_MIN_RELEVANCE from env.

        Raises EmbeddingModelMismatchError if the current model doesn't match.
        Returns:
            List of {text, source_file, page_number, chunk_type,
                     section_title, section_hierarchy, chapter, distance,
                     relevance_score, filtered_out_count}
        """
        meta = self._registry.get(equipment_id)
        if not meta:
            raise ValueError(f"Equipment '{equipment_id}' not registered.")

        # Guard: block queries if embedding model has changed
        self._check_embedding_compatibility(equipment_id)

        if min_relevance is None:
            min_relevance = DEFAULT_MIN_RELEVANCE

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

        # Fetch more than needed so we have room after filtering
        fetch_count = min(n_results * 2, collection.count()) if min_relevance > 0 else n_results
        fetch_count = min(fetch_count, collection.count())

        results = collection.query(
            query_texts=[query_text],
            n_results=fetch_count,
            where=where_filter,
        )

        formatted = []
        filtered_count = 0

        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta_item = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                relevance = self._l2_to_relevance(distance)

                # Filter out low-quality chunks
                if min_relevance > 0 and relevance < min_relevance:
                    filtered_count += 1
                    continue

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
                    "relevance_score": round(relevance, 1),
                })

                # Stop once we have enough high-quality results
                if len(formatted) >= n_results:
                    break

        if filtered_count > 0:
            logger.info(
                f"Retrieval filter: kept {len(formatted)}, "
                f"discarded {filtered_count} below {min_relevance}% relevance"
            )

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
