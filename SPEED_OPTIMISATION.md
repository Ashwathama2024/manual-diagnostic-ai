# ManualIQ — Speed Optimisation Reference

Researched: 2026-04-02 | Hardware target: 32 GB RAM, 16 logical CPUs, CPU-only (no GPU)

---

## Current Baseline

| Parameter | Value |
|---|---|
| Reasoning model | `deepseek-r1:8b` (Q4_K_M, ~8 GB RAM) |
| Embedding model | `nomic-embed-text` |
| Context window | 16 384 tokens |
| Output ceiling | 4 096 tokens |
| Threads | 12 |
| Est. CPU throughput | 8–14 t/s |

---

## Immediate Wins — No Model Change

These can be applied now in `config.yaml` and Windows system settings.

### 1 — Halve the context window (`config.yaml`)

```yaml
runtime:
  ollama_ctx_num: 8192      # was 16384
```

**Impact:** Biggest single speed gain available without touching the model.
The KV cache scales linearly with context length. Halving it frees ~1–2 GB RAM,
cuts prefill compute, and speeds generation. For chunked RAG retrieval, 8 192 tokens
is almost always sufficient — the 10 retrieved chunks + memory summary + history
fit comfortably.

### 2 — Increase threads to 14

```yaml
runtime:
  ollama_num_thread: 14     # was 12
```

On a 16-logical-CPU system, leaving 2 threads for OS/Ollama overhead is the
recommended starting point. Benchmark at 12, 14, and 16 to find your CPU's sweet spot.

### 3 — Ollama environment variables (Windows)

Set in **System Properties → Environment Variables** (or in the shell that starts Ollama):

```
OLLAMA_KEEP_ALIVE=60m
```
Keeps the model loaded between queries. Eliminates the 5–15 second cold-load
tax on every request. Critical for a RAG system with repeated queries.

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```
Quantises the KV cache from F16 → 8-bit (~50 % RAM reduction).
Requires `OLLAMA_FLASH_ATTENTION=1`. Frees RAM so model weights stay hot in cache.
Supported in Ollama v0.13.5+. Minimal quality impact.

```
OLLAMA_MAX_LOADED_MODELS=2
```
Required if running a two-model routing setup (see below). Keeps both models
warm simultaneously. Total RAM for phi4-mini + deepseek-r1:8b ≈ 13 GB — fits
comfortably in 32 GB.

### 4 — Windows Power Plan

Set to **High Performance** (or Ultimate Performance).
Laptop CPU throttling on Balanced / Battery-saver mode can halve inference speed.

---

## Quantisation Options for deepseek-r1:8b

You are already on Q4_K_M — the correct default for CPU-only inference.

| Quant | Disk | RAM | Relative Speed | Quality vs FP16 |
|---|---|---|---|---|
| **Q4_K_M** (current) | 5.2 GB | ~8 GB | 100 % (baseline) | ~96 % |
| Q5_K_M | 6.2 GB | ~9 GB | ~87 % | ~98 % |
| Q6_K | 7.4 GB | ~10 GB | ~77 % | ~99 % |
| Q8_0 | 8.9 GB | ~12 GB | ~67 % | ~99.5 % |
| FP16 | 16 GB | ~20 GB | ~45 % | 100 % |

**Verdict:** Q8_0 gives ~0.5 % quality improvement for a 33 % speed penalty.
Not worth it on CPU. Stay on Q4_K_M.

---

## Model Alternatives — Speed vs Reasoning Quality

| Model | Ollama Tag | Disk | RAM | Est. t/s | Reasoning | CoT Control |
|---|---|---|---|---|---|---|
| **phi4-mini** | `phi4-mini` | 2.8 GB | ~5 GB | 18–28 | ★★★★★ | Always-on |
| **qwen3:8b** | `qwen3:8b` | 5.2 GB | ~8 GB | 10–20 | ★★★★ | **Switchable** |
| deepseek-r1:8b *(current)* | `deepseek-r1:8b` | 5.2 GB | ~8 GB | 8–14 | ★★★★ | Always-on |
| deepseek-r1:7b | `deepseek-r1:7b` | 4.7 GB | ~7 GB | 12–18 | ★★★ | Always-on |
| qwen2.5:7b | `qwen2.5:7b` | 4.7 GB | ~7 GB | 14–20 | ★★ | None |
| llama3.1:8b | `llama3.1:8b` | 4.9 GB | ~7 GB | 14–20 | ★ | None |
| gemma3:4b | `gemma3:4b` | 3.2 GB | ~4 GB | 22–32 | ★ | None |

### Reasoning benchmarks

| Model | MATH-500 | GPQA Diamond | AIME 2024 |
|---|---|---|---|
| phi4-mini | **94.6 %** | **52.0 %** | 57.5 % |
| deepseek-r1-distill-llama-8b | 89.1 % | 49.0 % | **80.0 %** |
| qwen3:8b (think mode) | ~86 % | ~48 % | ~72 % |

### Recommended model upgrade: `phi4-mini`

- 3.8B parameters (half the size of R1-8b)
- Roughly **2× faster** on CPU
- **Scores higher** than R1-8b on MATH-500 and GPQA
- Weaker on AIME (pure maths olympiad) — irrelevant for technical manual RAG
- Pull: `ollama pull phi4-mini`
- Update `config.yaml`: `reasoning: phi4-mini`

Test with real Main Engine notebook queries before committing permanently.

---

## Two-Model Routing (Advanced)

The `THINK-GATE` classifier already in `server.py` can be extended to route
queries to different models, not just toggle CoT on/off:

```
Query → THINK-GATE classifier
  └─ Simple factual  →  phi4-mini          (~20 t/s, ~3 s response)
  └─ Complex analysis → deepseek-r1:8b     (~10 t/s, ~15 s response)
```

Both models fit in RAM simultaneously (~13 GB total).
Set `OLLAMA_MAX_LOADED_MODELS=2` to keep both warm.

This pattern covers ~70–80 % of typical notebook queries with the fast model,
reserving full chain-of-thought for synthesis and multi-step troubleshooting.

**Implementation sketch** (`server.py`):
```python
FAST_MODEL    = cfg["models"].get("fast_reasoning", "phi4-mini")
FULL_MODEL    = cfg["models"]["reasoning"]          # deepseek-r1:8b

model = FAST_MODEL if _is_simple_query(question) else FULL_MODEL
```

Add to `config.yaml`:
```yaml
models:
  reasoning:      deepseek-r1:8b
  fast_reasoning: phi4-mini        # used for simple factual lookups
```

---

## Speculative Decoding

Not yet natively supported in Ollama (tracked in GitHub issues #5800 and #9216).
The underlying llama.cpp engine supports it, but the Ollama interface hasn't
exposed it yet (as of April 2026). Monitor for future Ollama releases.

---

## Priority Order

| Priority | Action | Effort | Impact |
|---|---|---|---|
| 1 | Set `OLLAMA_KEEP_ALIVE=60m` env var | 2 min | High — eliminates cold-load latency |
| 2 | Reduce `ollama_ctx_num` 16384 → 8192 | 2 min | High — cuts KV cache, speeds prefill |
| 3 | Set `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` | 5 min | Medium — frees ~1–2 GB KV RAM |
| 4 | Set Windows Power Plan → High Performance | 1 min | Medium — prevents CPU throttle |
| 5 | Increase `num_thread` 12 → 14 | 2 min | Low–Medium — benchmark to confirm |
| 6 | Test `phi4-mini` as drop-in replacement | 30 min | High — ~2× speed, equal quality |
| 7 | Implement two-model routing in `server.py` | 2–4 h | High — best of both worlds |
