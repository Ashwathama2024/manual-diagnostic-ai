# ManualIQ — Upgrade Backlog

> Gaps found in current codebase (Phase 3 / 3.5 baseline).
> Grouped by theme. Each item has a short upgrade name, the gap it fixes, and effort rating.

*Last updated: 2026-04-01*

---

## 🚀 Speed Upgrades

---

### ⚡ CHUNK-CACHE
**Cache chunks.jsonl in memory per notebook**

| | |
|---|---|
| **File** | `query.py → load_chunks()` |
| **Gap** | `chunks.jsonl` is read from disk and fully parsed on every single query. A 500-chunk notebook = ~900 KB of JSON parsed fresh each time. In hybrid mode this effectively happens twice. |
| **Fix** | Module-level dict `{nb_id → [chunks]}`. Populated on first access, invalidated when a new file is uploaded to that notebook. |
| **Impact** | 🔴 High — biggest per-query I/O cost today |
| **Effort** | 🟢 Low — ~20 lines |

---

### ⚡ DB-PERSIST
**Keep LanceDB connection and embedder alive at startup**

| | |
|---|---|
| **File** | `query.py → retrieve_vector()` |
| **Gap** | `lancedb.connect()` and `OllamaEmbeddings()` are both instantiated fresh on every vector query — file handles, object state, and connection overhead every time. |
| **Fix** | Move both to module-level globals, initialised once in `server.py` startup alongside the Ollama client. Pass them into retrieval functions instead of re-creating. |
| **Impact** | 🟡 Medium — removes per-query connection overhead |
| **Effort** | 🟢 Low — ~15 lines |

---

### ⚡ PARALLEL-RETRIEVE
**Run BM25 and vector search concurrently in hybrid mode**

| | |
|---|---|
| **File** | `query.py → retrieve_hybrid()` |
| **Gap** | BM25 and vector searches run sequentially — vector only starts after BM25 finishes. They are completely independent operations on different data sources. |
| **Fix** | `concurrent.futures.ThreadPoolExecutor` with 2 workers; submit both tasks simultaneously, collect results with `as_completed`. |
| **Impact** | 🔴 High — roughly halves hybrid retrieval wall-clock time |
| **Effort** | 🟡 Medium — ~30 lines, needs thread-safety check on LanceDB |

---

### ⚡ BOOST-CACHE
**TTL in-memory cache for the boost table**

| | |
|---|---|
| **File** | `notebook_map.py → get_boost_table()` |
| **Gap** | Every query calls `get_boost_table()` which calls `load_map()` which does `path.read_text() + json.loads()`. The boost table only changes after `update_map()` (post-stream, background thread). |
| **Fix** | Module-level `{nb_id → (boost_table, timestamp)}`. Serve from cache if age < 60 s; reload on `update_map()` call. |
| **Impact** | 🟡 Medium — eliminates repeated JSON parse on every query |
| **Effort** | 🟢 Low — ~25 lines |

---

### ⚡ THINK-GATE
**Skip DeepSeek reasoning chain for simple factual lookups**

| | |
|---|---|
| **File** | `server.py → _stream_ollama()` |
| **Gap** | `think=True` is set on every query regardless of complexity. Simple lookups ("What is the FO pump diameter?") still run a 200–500 token internal reasoning chain, adding 15–50 s of invisible pre-output delay. |
| **Fix** | A lightweight query classifier (keyword signals: question length < 12 words, no "why/how/explain", answer likely a number or short fact) sets `think=False` for those calls. |
| **Impact** | 🟡 Medium — significant TTFT improvement on lookup queries |
| **Effort** | 🟡 Medium — ~40 lines for classifier + routing |

---

### ⚡ MEMORY-REGEN-LITE
**Use a lighter model for background memory regeneration**

| | |
|---|---|
| **File** | `notebook_map.py → regenerate_memory()` |
| **Gap** | Memory regen calls `_ollama_client.chat()` using the full DeepSeek-R1:8b model with `stream=False`. Ollama is single-queue — while this runs (~5–15 s), any new user query blocks waiting. |
| **Fix** | Add `memory_model` key to `config.yaml` (e.g. `llama3.2:3b` or `qwen2.5:3b`). Memory regen uses the lighter model; reasoning stays on DeepSeek. |
| **Impact** | 🟡 Medium — removes post-20th-query stall |
| **Effort** | 🟡 Medium — ~10 lines code + model pull |

---

## 🎯 Response Quality Upgrades

---

### 🎯 RRF-MERGE
**Reciprocal Rank Fusion for hybrid retrieval merge**

| | |
|---|---|
| **File** | `query.py → retrieve_hybrid()` |
| **Gap** | BM25 scores are unbounded floats (higher = better). Vector scores are L2 distances (lower = better). They are just concatenated with BM25 first — no normalisation, no proper merging. BM25 wins all deduplication ties. |
| **Fix** | Reciprocal Rank Fusion: for each chunk, `RRF = Σ 1 / (rank + 60)` across both result lists. Sort by RRF descending. Scale-agnostic, order-agnostic, consistently outperforms raw score concat. |
| **Impact** | 🔴 High — single biggest retrieval quality improvement available |
| **Effort** | 🟡 Medium — ~25 lines, replaces `dedupe_by_id` |

---

### 🎯 REAL-CITATIONS
**Parse [Chunk N] references from answer to filter true citations**

| | |
|---|---|
| **File** | `server.py → _chat_generator()` |
| **Gap** | All 10 retrieved chunks are emitted as citations regardless of whether the model referenced them. The sidebar shows 10 citations when the answer uses 2–3. `update_map()` records all 10 as "answered by", making the boost table noisy. |
| **Fix** | After streaming completes, regex-parse `full_answer` for `[Chunk N]` patterns. Filter `citations` to only the indices that actually appear. Pass filtered list to `update_map()` and the `done` SSE event. |
| **Impact** | 🟡 Medium — sharper boost table, cleaner sidebar, more honest citations |
| **Effort** | 🟢 Low — ~15 lines |

---

### 🎯 PARENT-CHUNK
**Fetch sibling chunks around top-k hits for richer context**

| | |
|---|---|
| **File** | `query.py` + `index.py` |
| **Gap** | A chunk retrieved for a maintenance procedure gives the LLM only that fragment. The surrounding steps, the safety warning above it, or the table header that contextualises it are not included. |
| **Fix** | At index time, store `prev_chunk_id` and `next_chunk_id` in each chunk record. At query time, for each top-k hit, fetch its siblings from the JSONL cache (CHUNK-CACHE prerequisite). Pass the expanded window to the LLM — small chunk for matching precision, large window for LLM context. |
| **Impact** | 🟡 Medium — measurably better answers on procedural and tabular content |
| **Effort** | 🟡 Medium — index schema change + retrieval expansion logic |

---

### 🎯 OVERLAP-400
**Increase chunk overlap from 200 → 400 characters**

| | |
|---|---|
| **File** | `config.yaml` |
| **Gap** | 200 chars (~2 lines) overlap is too small for numbered procedures. A 10-step maintenance procedure split across two chunks may have steps 6–7 in the overlap zone, with steps 1–5 and 8–10 on opposite sides. |
| **Fix** | Set `overlap_chars: 400` in config. Re-index all notebooks after change. Total chunk count increases ~10–15%, well within LanceDB and memory limits. |
| **Impact** | 🟡 Medium — fewer split procedures, more coherent chunks |
| **Effort** | 🟢 Low — 1 config line + re-index (automated) |

---

### 🎯 ABBREV-EXT
**Load user-defined abbreviations from a JSON sidecar file**

| | |
|---|---|
| **File** | `server.py → _ABBREV_MAP` |
| **Gap** | The abbreviation map has ~20 hardcoded marine terms. Users working on equipment with project-specific codes (e.g. "VIT", "FQS", "ECS") get no expansion benefit. Adding terms requires editing source code. |
| **Fix** | At startup, check for `data/abbreviations.json`. If present, merge into `_ABBREV_MAP`. Users edit the JSON file in any text editor. Document the format in README. |
| **Impact** | 🟡 Medium — retrieval quality improvement for domain-specific queries |
| **Effort** | 🟢 Low — ~10 lines |

---

### 🎯 EMBED-RERANK
**Second-pass embedding similarity rerank of top-k candidates**

| | |
|---|---|
| **File** | `query.py` + `server.py` |
| **Gap** | After hybrid merge and load-balancing, chunks are passed to the LLM in retrieval rank order. No second-pass semantic relevance check exists. A chunk that scores well lexically but is semantically off-topic can occupy a top slot. |
| **Fix** | After trimming to top_k, compute `cosine_similarity(query_embedding, chunk_embedding)` for each chunk using `nomic-embed-text` (already running). Re-sort by this score. The query embedding is already computed during vector search — cache and reuse it. |
| **Impact** | 🟡 Medium — better chunk ordering within the final set |
| **Effort** | 🟡 Medium — ~35 lines, query embedding reuse needed |

---

## 📋 Priority Order

| Priority | Upgrade | Reason |
|---|---|---|
| 1 | **CHUNK-CACHE** | Highest speed ROI, zero risk |
| 2 | **RRF-MERGE** | Highest quality ROI, no new dependencies |
| 3 | **DB-PERSIST** | Easy win, removes connection overhead |
| 4 | **BOOST-CACHE** | Pairs naturally with CHUNK-CACHE work |
| 5 | **REAL-CITATIONS** | Low effort, makes boost table accurate |
| 6 | **OVERLAP-400** | 1-line config change, re-index needed |
| 7 | **PARALLEL-RETRIEVE** | Halves hybrid latency, needs testing |
| 8 | **ABBREV-EXT** | Low effort, user-facing benefit |
| 9 | **THINK-GATE** | Good TTFT win, slightly more logic |
| 10 | **PARENT-CHUNK** | Index schema change, meaningful quality jump |
| 11 | **EMBED-RERANK** | Pairs well with PARENT-CHUNK |
| 12 | **MEMORY-REGEN-LITE** | Needs extra model pull, low-priority |

---

## 🗒️ Notes

- Items 1–5 can be implemented without any re-indexing of existing notebooks.
- **OVERLAP-400** and **PARENT-CHUNK** both require a full re-index after the change.
- **PARALLEL-RETRIEVE** depends on **DB-PERSIST** being done first (shared connection object must be thread-safe).
- **EMBED-RERANK** depends on **PARALLEL-RETRIEVE** or at minimum the query embedding being cached from the vector search step.
- None of these require new Ollama model downloads except **MEMORY-REGEN-LITE**.
