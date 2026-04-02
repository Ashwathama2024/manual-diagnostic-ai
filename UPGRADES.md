# ManualIQ - Upgrade Backlog

Gaps still worth addressing after the Phase 3 / 3.5 baseline.

Last updated: 2026-04-02

---

## Already completed on `main`

These items were previously listed as open, but the codebase already contains them:

| Upgrade | Status | Evidence |
|---|---|---|
| `CHUNK-CACHE` | Done | `load_chunks()` caches `chunks.jsonl` in memory per notebook |
| `DB-PERSIST` | Done | retrieval globals are pre-warmed and reused |
| `BOOST-CACHE` | Done | boost table has a TTL cache and invalidation on `update_map()` |
| `RRF-MERGE` | Done | hybrid retrieval uses Reciprocal Rank Fusion |
| `REAL-CITATIONS` | Done | streamed answer is parsed for `[Chunk N]` before persistence |

---

## Active backlog

### `THINK-GATE`
Skip heavy reasoning for simple factual lookups.

| | |
|---|---|
| File | `scripts/server.py` |
| Gap | Short lookup questions still pay the full reasoning delay. |
| Fix | Use a lightweight query classifier to set `think=False` for simple factual prompts. |
| Impact | High |
| Effort | Low |
| Status | Done on 2026-04-02 |

### `ABBREV-EXT`
Load user-defined abbreviations from a JSON sidecar file.

| | |
|---|---|
| File | `scripts/server.py` |
| Gap | Project-specific abbreviations require source edits today. |
| Fix | Merge optional `data/abbreviations.json` entries into the built-in abbreviation map at startup. |
| Impact | Medium |
| Effort | Low |
| Status | Done on 2026-04-02 |

### `OVERLAP-400`
Increase chunk overlap from 200 to 400 characters.

| | |
|---|---|
| File | `config.yaml` |
| Gap | Small overlap can split numbered procedures and table context too aggressively. |
| Fix | Set `overlap_chars: 400` and reindex notebooks. |
| Impact | Medium |
| Effort | Low |
| Status | Pending |

### `PARALLEL-RETRIEVE`
Run BM25 and vector retrieval concurrently in hybrid mode.

| | |
|---|---|
| File | `scripts/query.py` |
| Gap | Hybrid retrieval still performs lexical and vector passes sequentially. |
| Fix | Run both retrieval branches concurrently and fuse after both complete. |
| Impact | Medium |
| Effort | Medium |
| Status | Pending |

### `PARENT-CHUNK`
Expand top hits with neighboring chunks for richer context.

| | |
|---|---|
| File | `scripts/index.py`, `scripts/query.py` |
| Gap | Retrieved chunks can miss adjacent steps, warnings, or table headers. |
| Fix | Store sibling chunk references during indexing, then expand top hits during retrieval. |
| Impact | Medium |
| Effort | Medium |
| Status | Done on 2026-04-02 |

### `EMBED-RERANK`
Rerank the final candidate set with embedding similarity.

| | |
|---|---|
| File | `scripts/query.py` |
| Gap | Final chunk ordering still relies on retrieval rank only. |
| Fix | Reuse the query embedding and rerank the trimmed candidate set by semantic similarity. |
| Impact | Medium |
| Effort | Medium |
| Status | Pending |

### `MEMORY-REGEN-LITE`
Use a lighter model for notebook memory regeneration.

| | |
|---|---|
| File | `scripts/notebook_map.py`, `config.yaml` |
| Gap | Background memory regeneration can contend with interactive queries. |
| Fix | Add a smaller `memory_model` config and use it for periodic summaries. |
| Impact | Medium |
| Effort | Medium |
| Status | Pending |

---

## Recommended next order

| Priority | Upgrade | Reason |
|---|---|---|
| 1 | `OVERLAP-400` | Cheap quality win, but requires reindexing |
| 2 | `PARALLEL-RETRIEVE` | Useful latency win once retrieval path is stable |
| 3 | `EMBED-RERANK` | Good follow-up after retrieval ordering changes |
| 4 | `MEMORY-REGEN-LITE` | Nice-to-have, but lower urgency than retrieval work |

---

## Notes

- `THINK-GATE`, `ABBREV-EXT`, and `PARENT-CHUNK` are now implemented.
- `OVERLAP-400` requires a full reindex to take effect (sibling links are already stored by `PARENT-CHUNK`).
- `PARALLEL-RETRIEVE` should be tested carefully against shared retrieval objects.
