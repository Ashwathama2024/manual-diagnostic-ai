# ManualIQ — Competitive Gap Analysis & Product Roadmap

> **Comparison baseline:** AnythingLLM v1.7.x (Mintplex Labs, March 2026)
> **ManualIQ version:** Phase 1–2 complete (feature/phase1-2-core-knowledge branch)

---

## 1. Where ManualIQ Wins (Our Advantages)

These are genuine differentiators — features AnythingLLM either lacks entirely or implements generically.

| # | Capability | ManualIQ | AnythingLLM | Why It Matters |
|---|---|---|---|---|
| 1 | **Domain-aware prompt engineering** | Marine/industrial system prompt with strict anti-hallucination: fabricating pressures, torques, part codes is explicitly forbidden | Generic RAG prompt — no domain grounding | For maintenance troubleshooting, a hallucinated torque value can cause equipment damage or injury |
| 2 | **RAM-safe large PDF ingestion** | Docling page-batching (15 pages/batch) — 200 MB PDF never loaded into memory at once | Full PDF loaded by collector at once — crashes or hangs on image-heavy manuals | Shipboard manuals are 200–600 pages of scanned, image-heavy content |
| 3 | **Marine abbreviation expansion** | 21-term map: ME→main engine, FO→fuel oil, TC→turbocharger, etc. — baked into every retrieval query | None — queries pass through verbatim | Engineers type "ME FO pump pressure" — generic RAG finds nothing; ManualIQ understands |
| 4 | **Fuzzy typo-tolerant lexical retrieval** | Prefix + substring matching on BM25 scorer, 4-char minimum to avoid false positives | Pure vector similarity only | "crankshft", "turbocharger" misspelling — engineers work in noisy, stressful environments |
| 5 | **Dual knowledge hierarchy** | Core Knowledge Base (fundamental engineering theory, shared across all notebooks) + per-notebook manual sources | Flat workspace — all documents equal | Separate reusable reference library from vessel-specific manuals |
| 6 | **Equipment category taxonomy** | 11 marine machinery categories (propulsion, fuel oil, electrical, boiler, etc.) scoped to retrieval | None — workspaces are generic | Returns only hydraulically relevant chunks for a hydraulic query, not boiler manual chunks |
| 7 | **Hybrid retrieval with guaranteed per-source budgets** | vector + lexical combined; each source gets its own top-k budget so neither crowds out the other | Speed mode: pure vector only. Accuracy mode: vector + rerank (same pool) | Ensures core knowledge and manual both appear in context, neither silenced |
| 8 | **Section-path embedding strategy** | Chunks embedded as `[H1 > H2 > H3]\ntext` — topic context baked into the vector | Raw text embedded without structural context | A chunk from "Fuel Injection System > Pressure Adjustment" retrieves correctly even for vague queries |
| 9 | **DeepSeek reasoning transparency** | Native `think=True` streaming — collapsible "brain" panel shows the model's reasoning chain | No reasoning display — answer appears directly | Maintenance team can see WHY the system concluded a specific diagnosis; builds trust |
| 10 | **Watchdog auto-ingest for Core Knowledge** | Drop a PDF into `core_knowledge/` folder — system detects, ingests, and embeds automatically | Live sync requires configured connectors (GitHub, Confluence, etc.) | Zero-friction knowledge base maintenance for ship's engineer |

---

## 2. Where AnythingLLM Wins (Our Gaps)

These are features AnythingLLM ships that ManualIQ currently lacks.

### 2A — Critical Gaps (Block real-world multi-user deployment)

| # | Feature | AnythingLLM | ManualIQ Gap | Priority |
|---|---|---|---|---|
| G1 | **Multi-user / RBAC** | Admin / Manager / User roles; per-workspace user assignment | Single-user only — no login, no roles | 🔴 High — ships carry multiple engineers; Chief Engineer needs admin rights |
| G2 | **REST API** | Full CRUD + chat + upload API with scoped keys; Swagger docs | No external API — server.py only serves the SPA | 🔴 High — integration with ship management systems (PMS, CMMS) |
| G3 | **Reranking** | Accuracy-optimized mode re-scores top candidates before context injection | No reranking — pure score from BM25/vector distance | 🟡 Medium — would improve answer quality on complex multi-part questions |
| G4 | **Persistent chat history** | Chat history per workspace per user; survives browser reload | History lives in JS `State.history` — lost on page refresh | 🔴 High — engineers need to refer back to previous troubleshooting sessions |

### 2B — Document Formats & Connectors

| # | Feature | AnythingLLM | ManualIQ Gap | Priority |
|---|---|---|---|---|
| G5 | **Audio / video transcription** | Built-in Whisper (CPU); transcribes meetings, training videos | PDF and DOCX only | 🟡 Medium — safety briefing recordings, training videos |
| G6 | **XLSX / CSV ingestion** | Native | Not supported | 🟡 Medium — spare parts lists, maintenance logs |
| G7 | **PPTX ingestion** | Native | Not supported | 🟢 Low — manufacturer training decks |
| G8 | **Web scraper connector** | Single-page and recursive crawl | Not supported | 🟢 Low — manufacturer bulletins, class society circulars |
| G9 | **YouTube transcript ingestion** | One-click from URL | Not supported | 🟢 Low — OEM training channels |
| G10 | **Browser extension** | Chrome/Firefox — clip any webpage to workspace | Not supported | 🟢 Low |

### 2C — Agent & Automation Features

| # | Feature | AnythingLLM | ManualIQ Gap | Priority |
|---|---|---|---|---|
| G11 | **Autonomous agent mode** | `@agent` in chat triggers tools: web search, SQL, code exec, chart gen | Pure RAG only — no agentic loop | 🟡 Medium — could auto-query spare parts database |
| G12 | **Agent Flows (no-code)** | Visual pipeline builder — API call, file read/write, logic blocks | Not supported | 🟢 Low |
| G13 | **MCP server/client** | Full MCP compatibility — connect any MCP tool or expose ManualIQ to Claude Desktop | Not supported | 🟢 Low |
| G14 | **Code interpreter** | Execute Python/JS in sandbox from chat | Not supported | 🟢 Low |
| G15 | **SQL agent** | Natural language → SQL on connected databases | Not supported | 🟡 Medium — planned maintenance databases |

### 2D — Deployment & Ecosystem

| # | Feature | AnythingLLM | ManualIQ Gap | Priority |
|---|---|---|---|---|
| G16 | **Embedded chat widget** | `<script>` tag embeds workspace as chat bubble on any webpage | Not supported | 🟡 Medium — embed in vessel intranet portal |
| G17 | **Multiple LLM provider support** | 30+ providers — Ollama, OpenAI, Claude, Gemini, local GGUF | Ollama only | 🟢 Low — Ollama covers the offline use case |
| G18 | **Per-workspace LLM override** | Each workspace can use a different model | Global model only | 🟢 Low |
| G19 | **Multiple vector DB backends** | LanceDB, Chroma, PGVector, Qdrant, Milvus, Pinecone, Weaviate | LanceDB only | 🟢 Low — LanceDB is the right choice for offline |
| G20 | **Document versioning** | Live sync replaces (no diff); version tracking requested | No versioning | 🟡 Medium — track when manuals were updated |
| G21 | **Audit logging** | Basic admin panel logs | None | 🟡 Medium — ISM Code compliance, SMS requirements |

---

## 3. Feature Matrix Summary

```
Feature Area              ManualIQ    AnythingLLM
─────────────────────────────────────────────────
Offline / air-gapped         ✅✅         ✅
Domain-specific RAG          ✅✅         ❌
Technical document OCR       ✅✅         ✅ (basic)
RAM-safe large file ingest   ✅✅         ❌
Hybrid retrieval             ✅✅         ✅ (rerank only)
Fuzzy/typo tolerance         ✅          ❌
Reasoning transparency       ✅✅         ❌
Dual knowledge hierarchy     ✅✅         ❌
Equipment taxonomy           ✅✅         ❌
Live knowledge watcher       ✅          ✅ (connectors)
Multi-user / RBAC            ❌          ✅✅
Persistent chat history      ❌          ✅✅
REST API                     ❌          ✅✅
Agent / tool use             ❌          ✅✅
Multi-format (audio/xlsx)    ❌          ✅✅
Web connectors               ❌          ✅✅
Embedded widget              ❌          ✅
Document versioning          ❌          ❌ (partial)
Reranking                    ❌          ✅
SSO / SAML                   ❌          ❌ (both lack)
```

**Legend:** ✅✅ = clearly ahead, ✅ = adequate, ❌ = not present

---

## 4. Product Roadmap

Phases are ordered by impact vs. effort. Each builds on the previous.

---

### Phase 3 — Persistence & Multi-Session (Next Sprint)

*Goal: Stop losing work on browser refresh. Essential before sharing with a second person.*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 3.1 | **Persistent chat history** — save conversation turns to disk (`data/chats/{nb_id}.jsonl`); reload on notebook select | Small | 🔴 High |
| 3.2 | **Per-notebook system prompt override** — let user set a custom instruction per notebook (e.g., "always respond in metric units") | Small | 🟡 Medium |
| 3.3 | **Export conversation to PDF/MD** — download full chat session | Small | 🟡 Medium |
| 3.4 | **Document version tracking** — store ingest timestamp; show "last updated" next to each source | Small | 🟡 Medium |

---

### Phase 3.5 — Notebook Intelligence Layer *(Founding Vision — Original to ManualIQ)*

*Goal: Each notebook stops being a passive document store and becomes an active intelligence agent that learns from its own query history, builds an internal map of what it knows, and self-optimises over time. This is the architectural concept that separates ManualIQ from any general-purpose RAG system.*

> *"For notebooks processing substantial datasets, the system must autonomously generate a map for each notebook and query database — a reminder of the last query executed and the data sources utilised, providing a reference for subsequent similar queries. Maintaining a distinct history for each notebook aids the algorithm in sustaining an organised map and index, preventing intermingling of histories to ensure balanced retrieval loads."*
> — Product vision, ManualIQ founder, March 2026

#### 3.5.1 — Per-Notebook Query Map

Each notebook maintains a persistent `data/maps/{nb_id}_query_map.json` file updated after every chat turn. It records:

- **Query log** — every question asked, timestamp, retrieval mode used
- **Source utilisation** — which documents and chunk IDs were retrieved per query
- **Hot sections** — sections (by `section_path`) retrieved most frequently — the notebook's "known strong zones"
- **Cold sections** — document sections never queried — potential blind spots or coverage gaps
- **Topic clusters** — automatically grouped query terms (e.g., "fuel pump", "injection pressure", "FO viscosity" cluster together)

The map is a lightweight JSON — not a database — and survives server restarts. It is the notebook's long-term memory.

| ID | Sub-feature | Effort | Impact |
|---|---|---|---|
| 3.5.1a | Write query map JSON after every chat turn | Small | 🔴 Core |
| 3.5.1b | Track chunk utilisation frequency per source | Small | 🔴 Core |
| 3.5.1c | Auto-cluster query terms into topic groups | Medium | 🟡 Medium |
| 3.5.1d | Expose hot/cold section report in UI | Small | 🟡 Medium |

#### 3.5.2 — Learned Relevance Boost

Chunks that have been retrieved and cited in past answers receive a small score multiplier on future queries with similar terms. The system learns *which parts of the manual actually answer questions* vs. parts that get retrieved but ignored.

- Multiplier stored in the query map as `chunk_id → usage_count`
- Applied as a final re-weighting step after BM25 + vector scoring
- Decays slowly over time (prevents old queries from permanently dominating)
- Capped to avoid runaway amplification (max 2× boost)

| ID | Sub-feature | Effort | Impact |
|---|---|---|---|
| 3.5.2a | Track chunk retrieval frequency in query map | Small | 🔴 Core |
| 3.5.2b | Apply frequency multiplier in retrieval scorer | Small | 🔴 High |
| 3.5.2c | Implement time-decay on usage scores | Small | 🟡 Medium |

#### 3.5.3 — Notebook Memory Summary

After accumulating N queries, the LLM generates a concise natural-language summary of what the notebook is good at answering — stored as `data/maps/{nb_id}_memory.md`. This summary is:

- Injected into the system prompt as a "notebook personality" context block
- Updated automatically every 20 queries or on demand
- Readable by the user in the UI ("What does this notebook know?")

*Example output:* `"This notebook covers the MAN B&W 6S60MC-C main engine. Strong coverage: fuel injection timing, turbocharger maintenance, crankshaft bearing clearances. Limited coverage: governor settings, exhaust valve overhaul."`

| ID | Sub-feature | Effort | Impact |
|---|---|---|---|
| 3.5.3a | Generate memory summary via LLM after N queries | Medium | 🔴 High |
| 3.5.3b | Inject summary into system prompt as context | Small | 🔴 High |
| 3.5.3c | Show summary in UI "Notebook Intelligence" panel | Small | 🟡 Medium |
| 3.5.3d | Manual regenerate button for user | Tiny | 🟡 Medium |

#### 3.5.4 — Isolated Per-Notebook Chat History

Chat history is stored separately per notebook, never shared, never mixed.

- Storage: `data/chats/{nb_id}/sessions.jsonl` — append-only, one line per turn
- Each turn records: timestamp, question, answer, chunks cited, retrieval mode
- History feeds the query map (source of truth for topic clusters and chunk usage)
- History never influences retrieval in a *different* notebook — strict isolation
- UI shows past sessions per notebook; user can resume a previous session

| ID | Sub-feature | Effort | Impact |
|---|---|---|---|
| 3.5.4a | Write chat turns to per-notebook JSONL | Small | 🔴 Core |
| 3.5.4b | Load and display session history on notebook select | Medium | 🔴 High |
| 3.5.4c | Resume previous session (inject history into context) | Small | 🟡 Medium |
| 3.5.4d | Session list with timestamps in sidebar | Small | 🟡 Medium |
| 3.5.4e | Delete / archive individual sessions | Small | 🟢 Low |

#### 3.5.5 — Retrieval Load Balancing

When a notebook contains many sources (e.g., 10 manuals), a naive retriever lets large documents crowd out small ones — a 500-page manual drowns a 20-page technical bulletin. Load balancing ensures proportional representation.

- **Source-proportional chunk budget**: each source gets a chunk budget proportional to its size (chunks per source / total chunks), scaled to top-k
- **Minimum floor**: every selected source gets at least 1 chunk in context, regardless of size
- **Query-type routing**: if the query map identifies the query as belonging to a known topic cluster, boost the sources most associated with that cluster
- **Adaptive top-k**: for large notebooks (>500 chunks), dynamically increase top-k to maintain coverage without degrading latency

| ID | Sub-feature | Effort | Impact |
|---|---|---|---|
| 3.5.5a | Compute per-source chunk budgets at retrieval time | Medium | 🔴 High |
| 3.5.5b | Enforce minimum 1-chunk floor per selected source | Small | 🟡 Medium |
| 3.5.5c | Query-type-to-source affinity routing from query map | Large | 🟡 Medium |
| 3.5.5d | Adaptive top-k for large notebooks | Small | 🟡 Medium |

---

### Phase 4 — REST API + Integration Layer

*Goal: Allow external systems (PMS, CMMS, SMS portals) to query ManualIQ programmatically.*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 4.1 | **Public REST API** — `/api/v1/chat`, `/api/v1/notebooks`, `/api/v1/upload` with API key auth | Medium | 🔴 High |
| 4.2 | **API key management UI** — generate/revoke keys in settings panel | Small | 🔴 High |
| 4.3 | **Swagger / OpenAPI docs** — auto-generated at `/api/docs` | Tiny | 🟡 Medium |
| 4.4 | **Webhook on ingest complete** — POST to configurable URL when new document is indexed | Small | 🟡 Medium |

---

### Phase 5 — Multi-User & Access Control

*Goal: Deploy on ship's LAN so Chief Engineer, 1st Engineer, and crew all have scoped access.*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 5.1 | **User accounts** — local username/password; JWT session tokens | Medium | 🔴 High |
| 5.2 | **Roles: Admin / Engineer / Viewer** — Admin manages system; Engineer uploads/queries; Viewer queries only | Medium | 🔴 High |
| 5.3 | **Notebook-level access control** — assign notebooks to specific users or roles | Medium | 🟡 Medium |
| 5.4 | **Audit log** — record who queried what, when; exportable CSV (ISM/SMS compliance) | Medium | 🟡 Medium |
| 5.5 | **Multi-user chat isolation** — separate conversation history per user per notebook | Small | 🟡 Medium |

---

### Phase 6 — Retrieval Quality Upgrades

*Goal: Improve answer accuracy, especially for complex multi-step troubleshooting queries.*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 6.1 | **Reranker model integration** — second-pass reranking of top-20 candidates using cross-encoder (e.g., `bge-reranker-v2-m3` via Ollama) | Medium | 🔴 High |
| 6.2 | **Query decomposition** — break multi-part questions into sub-queries, retrieve independently, merge context | Medium | 🔴 High |
| 6.3 | **Contextual chunk expansion** — when a chunk scores high, include the ±1 surrounding chunks for continuity | Small | 🟡 Medium |
| 6.4 | **Metadata-filtered retrieval** — filter by equipment category, date range, or document title from query | Medium | 🟡 Medium |
| 6.5 | **Expand abbreviation dictionary** — from 21 terms to 100+ (IMO, MARPOL, SOLAS, ISM standard terms) | Small | 🟡 Medium |

---

### Phase 7 — Document Format Expansion

*Goal: Ingest the full breadth of shipboard documentation formats.*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 7.1 | **XLSX / CSV ingestion** — parse spreadsheets as structured markdown tables; preserve column headers | Medium | 🟡 Medium — spare parts lists, maintenance schedules |
| 7.2 | **PPTX ingestion** — extract slide text and speaker notes | Small | 🟢 Low — OEM training decks |
| 7.3 | **Audio transcription** — Whisper via Ollama; ingest safety briefings, recorded toolbox talks | Medium | 🟡 Medium |
| 7.4 | **Scanned image / photo ingestion** — OCR on standalone JPEG/PNG (e.g., photo of a nameplate) | Small | 🟡 Medium |
| 7.5 | **P&ID / schematic understanding** — vision model describes piping & instrumentation diagrams; extract tagged component IDs | Large | 🔴 High — core gap vs. specialist systems |

---

### Phase 8 — Agentic Capabilities

*Goal: Move from "answer questions about documents" to "help solve problems autonomously".*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 8.1 | **Spare parts lookup agent** — query maker website or local parts database by part number from context | Medium | 🔴 High |
| 8.2 | **Maintenance history agent** — integrate with PMS (Planned Maintenance System) via REST; query past maintenance records | Large | 🔴 High |
| 8.3 | **Alarm interpretation agent** — map active alarms (from AMS/IAS system) to relevant manual sections automatically | Large | 🔴 High |
| 8.4 | **Checklist generator** — LLM-generated step-by-step checklists from procedure sections, exportable as PDF | Medium | 🟡 Medium |
| 8.5 | **Web search agent** — optional online lookup for class society circulars, manufacturer bulletins | Small | 🟢 Low |
| 8.6 | **MCP server** — expose ManualIQ workspaces to Claude Desktop, TypingMind, or custom tools via MCP protocol | Medium | 🟢 Low |

---

### Phase 9 — UX & Polish

*Goal: Professional-grade UI ready for shipboard deployment without IT support.*

| ID | Feature | Effort | Impact |
|---|---|---|---|
| 9.1 | **Search across all notebooks** — global search before selecting a notebook | Small | 🟡 Medium |
| 9.2 | **Document page preview** — click a citation to see the original PDF page in-browser | Medium | 🟡 Medium |
| 9.3 | **Embeddable chat widget** — `<iframe>` snippet for vessel intranet portals | Medium | 🟡 Medium |
| 9.4 | **Mobile / tablet layout** — responsive breakpoints for bridge iPad use | Medium | 🟡 Medium |
| 9.5 | **Offline installation wizard** — one-click setup: check Ollama, pull models, verify LanceDB | Small | 🔴 High — deployment without IT |
| 9.6 | **Dark / light mode toggle** | Small | 🟢 Low |

---

## 5. What ManualIQ Should Selectively Borrow from AnythingLLM

> We are a domain expert, not a general platform. This table identifies the small subset of AnythingLLM features that are genuinely useful for marine/industrial deployment — and explicitly rejects the rest.

### ✅ Borrow — High Value, Low Distraction

| Feature | Why it fits ManualIQ | Planned Phase |
|---|---|---|
| **Persistent chat history per workspace** | Engineers reference yesterday's troubleshooting session; isolated per notebook | Phase 3.5.4 |
| **REST API with scoped keys** | Integration with PMS, CMMS, SMS portals; automation pipelines | Phase 4 |
| **Role-based access control (Admin / Engineer / Viewer)** | Chief Engineer manages system; crew queries only | Phase 5 |
| **Cross-encoder reranking** | Improves multi-part technical queries; single model, offline | Phase 6.1 |
| **XLSX / CSV ingestion** | Spare parts lists, maintenance schedules, calibration records | Phase 7.1 |
| **Embedded chat widget** | Vessel intranet portal deployment without opening a new tool | Phase 9.3 |
| **Document page preview** | Click a citation, see the exact PDF page — critical for verification | Phase 9.2 |

### 🟡 Borrow Later — Medium Value, Moderate Effort

| Feature | Condition for adoption |
|---|---|
| **Audio transcription (Whisper)** | Only if shipboard safety briefings and toolbox talks are a confirmed use case |
| **Scanned image / photo ingestion** | Useful for nameplate OCR; adopt when Docling vision pipeline is already GPU-enabled |
| **Audit logging / export** | Required before any ISM Code or class society submission use case |

### ❌ Do Not Borrow — Wrong Fit for a Domain Specialist

| Feature | Why to skip |
|---|---|
| **30+ LLM provider support** | Ollama covers the offline, air-gapped requirement. Adding cloud APIs introduces network dependency and a potential data leak surface |
| **No-code Agent Flows** | Our agents must have domain logic (alarm-to-manual mapping, spare parts lookup). Generic drag-and-drop flows produce generic agents |
| **Community Hub / shared skills** | A vessel's knowledge base is confidential — ISM code, class survey records, technical bulletins are not community-shareable |
| **GitHub / GitLab / Confluence connectors** | Ship does not run a code repository or a Confluence wiki; zero shipboard relevance |
| **YouTube transcript ingestion** | Very low reliability for technical accuracy; OEM videos contain imprecise language |
| **Cloud vector databases (Pinecone, Weaviate, Zilliz)** | Offshore and at-sea operations have no internet; LanceDB on local NVMe is the correct architecture |
| **Web scraper connector** | Unreliable for structured technical content; class society documents require login-gated access anyway |
| **PPTX ingestion** | Low priority; manufacturer decks are mostly marketing, not maintenance procedure |
| **Per-workspace LLM override** | Complexity without benefit; one well-tuned DeepSeek-R1 serves all notebooks better than per-notebook model chaos |
| **MCP server / client** | Premature — no MCP client ecosystem exists on a vessel; revisit in 2027 if marine ERP vendors adopt MCP |
| **Light mode** | Dark mode is correct for engine room and bridge environments; light mode is a distraction |

---

## 7. Strategic Positioning Summary

ManualIQ and AnythingLLM are **not the same product**.

AnythingLLM is a **horizontal general-purpose RAG platform** — it aims to work for any document, any domain, any team. Its strength is breadth: 30+ LLM providers, 9 vector DBs, agents, connectors, no-code flows.

ManualIQ is a **vertical specialist** — built from the ground up for marine and industrial engineering documentation. Its strength is depth: it understands what "ME FO pressure drop" means, it can handle a 400-page scanned engine room manual without crashing, and it will never tell an engineer that a relief valve cracks at a pressure it fabricated.

**The correct competitive move is NOT to catch up to AnythingLLM on breadth.** The correct move is to:

1. **Defend the domain moat** — deepen engineering intelligence (abbreviations, P&ID parsing, alarm cross-referencing, class society rule integration)
2. **Close the deployment gaps** — multi-user, persistent history, REST API (Phases 3–5)
3. **Add targeted agentic features** — not general agents, but *marine-specific* ones (spare parts lookup, PMS integration, alarm-to-manual mapping)
4. **Ship as an offline-first appliance** — one-command install, no cloud dependency, runs on a ruggedized laptop on a vessel at sea

The market AnythingLLM cannot easily enter — oil rigs, cargo vessels, navy vessels, offshore platforms, ATEX zones — is exactly where ManualIQ should win.

---

*Last updated: 2026-03-31*
*Gap analysis by: Claude Sonnet 4.6*
*Notebook Intelligence Layer vision: ManualIQ founder*
