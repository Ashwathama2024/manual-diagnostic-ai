"""
LLM Reasoning Engine — Local Ollama Integration
=================================================
Connects to a locally running Ollama server to provide
AI-powered diagnostic reasoning based on retrieved manual data.

The engine:
  1. Takes user question + retrieved context chunks (with section metadata)
  2. Builds a structured prompt with section-aware citations
  3. Sends to local LLM via Ollama (streaming or full)
  4. Returns formatted diagnostic response with manual references

Runs 100% offline — no cloud API calls.
"""

import os
import logging
import threading
import time
from typing import Optional, Generator
from queue import Queue, Empty

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.3:8b")

# ---------------------------------------------------------------------------
# Timeout & Fallback Configuration
# ---------------------------------------------------------------------------
# Time (seconds) to wait for the FIRST token from LLM.
# CPU inference can be slow for large models — default 120s is generous.
LLM_FIRST_TOKEN_TIMEOUT = int(os.environ.get("LLM_FIRST_TOKEN_TIMEOUT", "120"))

# Time (seconds) allowed between consecutive tokens during streaming.
# If no new token arrives within this window, treat it as a stall.
LLM_INTER_TOKEN_TIMEOUT = int(os.environ.get("LLM_INTER_TOKEN_TIMEOUT", "60"))

# Total wall-clock time (seconds) for the entire response.
# Prevents runaway generation. 0 = no limit.
LLM_MAX_RESPONSE_TIME = int(os.environ.get("LLM_MAX_RESPONSE_TIME", "600"))

# Fallback model — used automatically if the primary model times out or errors.
# Set to empty string to disable fallback.
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "")

# Maximum fallback attempts before giving up.
MAX_FALLBACK_ATTEMPTS = int(os.environ.get("MAX_FALLBACK_ATTEMPTS", "1"))


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class LLMTimeoutError(Exception):
    """Raised when LLM inference exceeds timeout thresholds."""
    def __init__(self, message: str, timeout_type: str = "unknown", elapsed: float = 0):
        super().__init__(message)
        self.timeout_type = timeout_type  # "first_token", "inter_token", "total"
        self.elapsed = elapsed


class LLMStallError(Exception):
    """Raised when LLM stops producing tokens mid-response."""
    def __init__(self, message: str, partial_response: str = ""):
        super().__init__(message)
        self.partial_response = partial_response


class LLMCancelledError(Exception):
    """Raised when the user or system cancels an in-flight inference."""
    pass


# ---------------------------------------------------------------------------
# Recommended models — user picks based on their hardware
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS = {
    "llama3.3:8b": {
        "name": "Llama 3.3 8B",
        "ram": "16 GB",
        "vram": "8 GB",
        "quality": "Excellent",
        "speed": "Fast",
        "description": "Best overall — strong reasoning, fast on 16GB RAM or 8GB VRAM GPU",
    },
    "qwen2.5:7b": {
        "name": "Qwen 2.5 7B",
        "ram": "16 GB",
        "vram": "8 GB",
        "quality": "Excellent",
        "speed": "Fast",
        "description": "Top-tier for technical reasoning and structured output",
    },
    "mistral:7b": {
        "name": "Mistral 7B",
        "ram": "16 GB",
        "vram": "8 GB",
        "quality": "Very Good",
        "speed": "Fast",
        "description": "Reliable workhorse — strong summarization and analysis",
    },
    "gemma2:9b": {
        "name": "Gemma 2 9B",
        "ram": "16 GB",
        "vram": "10 GB",
        "quality": "Excellent",
        "speed": "Moderate",
        "description": "Google's best open model — excellent at following instructions",
    },
    "phi3:3.8b": {
        "name": "Phi-3 3.8B",
        "ram": "8 GB",
        "vram": "4 GB",
        "quality": "Good",
        "speed": "Very Fast",
        "description": "Lightweight champion — runs on 8GB RAM, punches above its weight",
    },
    "llama3.3:70b": {
        "name": "Llama 3.3 70B",
        "ram": "48 GB",
        "vram": "24 GB (Q4)",
        "quality": "Best",
        "speed": "Slow",
        "description": "Maximum quality — needs 48GB+ RAM or 24GB+ VRAM GPU",
    },
    "qwen2.5:72b": {
        "name": "Qwen 2.5 72B",
        "ram": "48 GB",
        "vram": "24 GB (Q4)",
        "quality": "Best",
        "speed": "Slow",
        "description": "Top reasoning at 72B scale — rivals GPT-4 on technical tasks",
    },
    "deepseek-r1:8b": {
        "name": "DeepSeek R1 8B",
        "ram": "16 GB",
        "vram": "8 GB",
        "quality": "Excellent",
        "speed": "Fast",
        "description": "Strong chain-of-thought reasoning — great for diagnostics",
    },
    "command-r:35b": {
        "name": "Command R 35B",
        "ram": "32 GB",
        "vram": "16 GB",
        "quality": "Excellent",
        "speed": "Moderate",
        "description": "Built for RAG — excels at citing sources and retrieval tasks",
    },
}


# ---------------------------------------------------------------------------
# System prompt for diagnostic reasoning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Senior Marine / Industrial Equipment Diagnostic Engineer with 30+ years of experience. You work EXCLUSIVELY from the technical manual data provided to you — never guess or use general knowledge.

## Your Operating Principles

1. **Manual-First**: Every statement must be traceable to the provided manual excerpts.
   If the manual doesn't cover the topic, say: "The uploaded manuals do not contain information about this topic."

2. **Engineering Reasoning**: For every diagnosis, explain the engineering WHY:
   - What physical principle causes this symptom?
   - What is the causal chain from root cause to observable effect?
   - What are the thermodynamic / mechanical / electrical relationships?

3. **Structured Diagnostics**: Follow this diagnostic framework:
   a) SYMPTOM ANALYSIS — What exactly is the operator observing?
   b) POSSIBLE CAUSES — Ranked by probability (from manual troubleshooting data)
   c) DIAGNOSTIC STEPS — Step-by-step checks, referencing manual procedures
   d) CORRECTIVE ACTIONS — Specific repairs/adjustments per the manual
   e) PREVENTIVE MEASURES — How to avoid recurrence

4. **Safety First**: Always highlight safety warnings from the manual.
   If a procedure involves hazards, state them prominently.

5. **Precise Source Citations**: Reference the EXACT manual location for every key claim.
   Use this format: **[Source: {manual filename} > {section hierarchy} > Page {X}]**
   Example: **[Source: MAN_B&W_Manual.pdf > Chapter 3: Fuel System > 3.2.1 Injection Timing > Page 45]**

6. **Quantitative**: Use specific values from the manual — tolerances, clearances,
   pressures, temperatures — never vague statements like "check if it's too hot."

## Response Format

When answering diagnostic questions, structure your response as:

### Symptom Analysis
[Description of the reported condition]

### Probable Causes
1. [Most likely] — [engineering reasoning why]
   **[Source: manual > section > page]**
2. [Next likely] — [engineering reasoning why]
   **[Source: manual > section > page]**

### Diagnostic Procedure
Step 1: [Action] — **[Source: manual > section > page]**
Step 2: [Action] — **[Source: manual > section > page]**

### Corrective Action
[Specific repair steps from the manual]

### Safety Notes
[Any relevant warnings or precautions from the manual]

---
For general information questions (not diagnostics), provide clear, accurate answers
citing the manual section and page. Keep the engineering depth appropriate to the question."""


# ---------------------------------------------------------------------------
# Context builder — section-aware
# ---------------------------------------------------------------------------

def build_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM.
    Includes full section hierarchy for precise citations.
    Adds relevance quality tier so LLM can weigh evidence appropriately.
    """
    if not retrieved_chunks:
        return "No relevant manual data found for this query."

    context_parts = []
    context_parts.append("=" * 60)
    context_parts.append("RETRIEVED MANUAL DATA (use ONLY this data to answer)")
    context_parts.append(f"({len(retrieved_chunks)} excerpts passed quality filter)")
    context_parts.append("=" * 60)

    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source_file", "Unknown")
        page = chunk.get("page_number", "?")
        ctype = chunk.get("chunk_type", "text")

        # Use pre-computed relevance score if available (accurate L2→cosine),
        # otherwise fall back to simple distance conversion
        if "relevance_score" in chunk:
            relevance = chunk["relevance_score"]
        else:
            distance = chunk.get("distance", 0)
            relevance = max(0, round((1 - distance / 2.0) * 100, 1))

        # Quality tier for LLM awareness
        if relevance >= 70:
            tier = "HIGH"
        elif relevance >= 50:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        # Build the reference path
        section = chunk.get("section_hierarchy", "") or chunk.get("section_title", "")
        chapter = chunk.get("chapter", "")

        ref_parts = [f"Source: {source}"]
        if chapter:
            ref_parts.append(chapter)
        if section and section != chapter:
            ref_parts.append(section)
        ref_parts.append(f"Page {page}")
        reference = " > ".join(ref_parts)

        context_parts.append(
            f"\n--- Excerpt {i} [{ctype.upper()}] ({reference}) "
            f"[Relevance: {relevance}% — {tier}] ---"
        )
        context_parts.append(chunk.get("text", ""))

    context_parts.append("\n" + "=" * 60)
    context_parts.append("CITATION INSTRUCTION: When referencing this data, cite as:")
    context_parts.append("**[Source: filename > section hierarchy > Page X]**")
    context_parts.append("Prioritize HIGH relevance excerpts. Use LOW relevance excerpts only if they directly support your analysis.")
    context_parts.append("=" * 60)
    return "\n".join(context_parts)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def check_ollama_status() -> dict:
    """Check if Ollama is running and what models are available."""
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        models = client.list()
        model_names = []
        if hasattr(models, 'models'):
            model_names = [m.model for m in models.models]
        elif isinstance(models, dict) and 'models' in models:
            model_names = [m.get('name', m.get('model', '')) for m in models['models']]

        return {
            "running": True,
            "models": model_names,
            "url": OLLAMA_BASE_URL,
        }
    except Exception as e:
        return {
            "running": False,
            "error": str(e),
            "url": OLLAMA_BASE_URL,
        }


def get_available_models() -> list[str]:
    """Get list of models available in Ollama."""
    status = check_ollama_status()
    return status.get("models", [])


def _build_user_message(
    question: str,
    retrieved_chunks: list[dict],
    equipment_name: str = "",
    conversation_context: str = "",
) -> str:
    """
    Build the user prompt with context, conversation history, and question.

    Args:
        question: The user's current question
        retrieved_chunks: Relevant chunks from vector search
        equipment_name: Active equipment name
        conversation_context: Formatted conversation history from ConversationMemory
    """
    context = build_context(retrieved_chunks)
    equipment_context = ""
    if equipment_name:
        equipment_context = f"\nYou are answering questions about: **{equipment_name}**\n"

    # Inject conversation history if available
    conv_block = ""
    if conversation_context:
        conv_block = f"""
{conversation_context}

---
"""

    return f"""{equipment_context}{conv_block}
## User Question
{question}

## Manual Data
{context}

Provide your analysis based STRICTLY on the manual data above. Cite every key claim with **[Source: filename > section > Page X]**. If the data doesn't contain relevant information, clearly state that. If this is a follow-up question, use the conversation history above to understand what the user is referring to."""


def _stream_ollama_to_queue(
    client,
    model: str,
    messages: list[dict],
    queue: Queue,
    cancel_event: threading.Event,
):
    """
    Worker thread: runs Ollama streaming chat and pushes tokens to a Queue.
    Pushes None as sentinel when done, or an Exception on failure.
    Checks cancel_event between tokens — exits early if set.
    """
    try:
        response_stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in response_stream:
            if cancel_event.is_set():
                queue.put(LLMCancelledError("Inference cancelled by user/system"))
                return

            token = None
            if isinstance(chunk, dict) and "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
            elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                token = chunk.message.content
            else:
                logger.warning(f"Unexpected chunk format: {chunk}")
                continue

            if token is not None:
                queue.put(token)

        queue.put(None)  # sentinel: stream complete
    except Exception as e:
        queue.put(e)


def generate_response(
    question: str,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    equipment_name: str = "",
    stream: bool = True,
    cancel_event: threading.Event = None,
    first_token_timeout: int = None,
    inter_token_timeout: int = None,
    max_response_time: int = None,
    conversation_context: str = "",
) -> Generator[str, None, None]:
    """
    Generate a diagnostic response using the local LLM — STREAMING with timeout.

    Args:
      conversation_context: Formatted conversation history from ConversationMemory.
                           Injected into the user message so LLM can resolve references.

    Timeout behaviour:
      - first_token_timeout: max seconds to wait for the first token (default: LLM_FIRST_TOKEN_TIMEOUT)
      - inter_token_timeout: max seconds gap between consecutive tokens (default: LLM_INTER_TOKEN_TIMEOUT)
      - max_response_time: total wall-clock budget for the entire response (default: LLM_MAX_RESPONSE_TIME, 0=unlimited)

    Cancellation:
      - Pass a threading.Event as cancel_event; set it externally to abort inference.

    Raises:
      - LLMTimeoutError on any timeout
      - LLMCancelledError on cancellation
      - ConnectionError / other ollama errors propagate as-is
    """
    import ollama

    # Resolve timeouts
    ft_timeout = first_token_timeout if first_token_timeout is not None else LLM_FIRST_TOKEN_TIMEOUT
    it_timeout = inter_token_timeout if inter_token_timeout is not None else LLM_INTER_TOKEN_TIMEOUT
    total_timeout = max_response_time if max_response_time is not None else LLM_MAX_RESPONSE_TIME

    if cancel_event is None:
        cancel_event = threading.Event()

    user_message = _build_user_message(question, retrieved_chunks, equipment_name, conversation_context)

    client = ollama.Client(host=OLLAMA_BASE_URL)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        f"Querying Ollama model '{model}' with {len(retrieved_chunks)} context chunks "
        f"[timeouts: first_token={ft_timeout}s, inter_token={it_timeout}s, total={total_timeout}s]"
    )

    # Launch Ollama streaming in a background thread
    token_queue: Queue = Queue()
    worker = threading.Thread(
        target=_stream_ollama_to_queue,
        args=(client, model, messages, token_queue, cancel_event),
        daemon=True,
    )
    worker.start()

    start_time = time.monotonic()
    first_token_received = False
    partial_response = ""
    token_count = 0

    while True:
        # Determine which timeout applies right now
        if not first_token_received:
            current_timeout = ft_timeout
            timeout_label = "first_token"
        else:
            current_timeout = it_timeout
            timeout_label = "inter_token"

        # Check total wall-clock limit
        elapsed = time.monotonic() - start_time
        if total_timeout > 0 and elapsed >= total_timeout:
            cancel_event.set()  # signal worker to stop
            raise LLMTimeoutError(
                f"Total response time exceeded {total_timeout}s "
                f"(generated {token_count} tokens in {elapsed:.1f}s)",
                timeout_type="total",
                elapsed=elapsed,
            )

        # Wait for next token from the queue
        try:
            item = token_queue.get(timeout=current_timeout)
        except Empty:
            elapsed = time.monotonic() - start_time
            cancel_event.set()  # signal worker to stop
            if not first_token_received:
                raise LLMTimeoutError(
                    f"No response from model '{model}' after {ft_timeout}s. "
                    f"The model may be loading or the system is overloaded.",
                    timeout_type="first_token",
                    elapsed=elapsed,
                )
            else:
                raise LLMTimeoutError(
                    f"Model '{model}' stalled — no token for {it_timeout}s "
                    f"(after {token_count} tokens, {elapsed:.1f}s elapsed). "
                    f"Partial response was generated.",
                    timeout_type="inter_token",
                    elapsed=elapsed,
                )

        # Handle sentinel (stream complete)
        if item is None:
            break

        # Handle exceptions from worker thread
        if isinstance(item, Exception):
            raise item

        # Handle cancellation check
        if cancel_event.is_set():
            raise LLMCancelledError("Inference cancelled")

        # It's a token — yield it
        first_token_received = True
        token_count += 1
        partial_response += item
        yield item

    elapsed = time.monotonic() - start_time
    logger.info(
        f"Response complete: {token_count} tokens in {elapsed:.1f}s "
        f"({token_count/max(elapsed, 0.1):.1f} tok/s) from '{model}'"
    )



def generate_response_full(
    question: str,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    equipment_name: str = "",
    timeout: int = None,
) -> str:
    """
    Non-streaming version: returns the complete response as a string.
    Uses the streaming generator internally to benefit from timeout protection.
    """
    total_timeout = timeout if timeout is not None else LLM_MAX_RESPONSE_TIME
    tokens = []
    for token in generate_response(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model=model,
        equipment_name=equipment_name,
        max_response_time=total_timeout,
    ):
        tokens.append(token)

    return "".join(tokens) if tokens else "No response generated."


def generate_response_with_fallback(
    question: str,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    equipment_name: str = "",
    cancel_event: threading.Event = None,
    fallback_model: str = None,
    conversation_context: str = "",
) -> Generator[str, None, None]:
    """
    Streaming response with automatic fallback to a smaller/faster model on timeout.

    Behaviour:
      1. Try the primary model with configured timeouts.
      2. On LLMTimeoutError → yield a warning message → retry with fallback_model.
      3. If fallback also fails → raise the error.

    The caller sees a seamless token stream — the fallback switch is transparent
    except for an inline status message.
    """
    fb_model = fallback_model or FALLBACK_MODEL
    models_to_try = [model]
    if fb_model and fb_model != model:
        models_to_try.append(fb_model)

    last_error = None

    for attempt, current_model in enumerate(models_to_try):
        try:
            if attempt > 0:
                # Yield inline warning so the user sees the fallback in chat
                yield f"\n\n---\n*Primary model timed out. Switching to fallback: `{current_model}`*\n\n---\n\n"
                logger.warning(f"Falling back to model '{current_model}' after timeout on '{model}'")

            for token in generate_response(
                question=question,
                retrieved_chunks=retrieved_chunks,
                model=current_model,
                equipment_name=equipment_name,
                cancel_event=cancel_event,
                conversation_context=conversation_context,
            ):
                yield token
            return  # success — exit

        except LLMTimeoutError as e:
            last_error = e
            logger.error(f"Timeout on model '{current_model}': {e} (type={e.timeout_type}, elapsed={e.elapsed:.1f}s)")
            if attempt >= len(models_to_try) - 1:
                raise  # no more fallbacks
            continue

        except LLMCancelledError:
            raise  # user cancelled — don't fallback

        except Exception:
            raise  # connection errors etc — don't fallback


# ---------------------------------------------------------------------------
# Conversation memory (per-session)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Maintains conversation context for multi-turn diagnostic conversations.

    Key capabilities:
      1. Stores recent Q&A exchanges with source references
      2. Detects follow-up questions (pronouns, short queries, implicit references)
      3. Expands follow-up queries with context for better retrieval
      4. Provides conversation history for LLM prompt injection
    """

    # Words that suggest a follow-up rather than a new topic
    FOLLOW_UP_SIGNALS = {
        # Pronouns referencing prior context
        "it", "its", "this", "that", "these", "those", "they", "them",
        "the same", "the above", "the previous", "mentioned",
        # Conversation continuers
        "also", "what about", "how about", "and the", "regarding",
        "what if", "but what", "so what", "then what",
        # Short imperative follow-ups
        "explain", "elaborate", "clarify", "expand", "detail",
        "why", "how",
    }

    def __init__(self, max_history: int = 10):
        self.history: list[dict] = []
        self.max_history = max_history

    def add_exchange(self, question: str, answer: str, sources: list[dict] = None):
        self.history.append({
            "question": question,
            "answer": answer[:500],  # cap stored answer length
            "sources": sources or [],
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def is_follow_up(self, query: str) -> bool:
        """
        Detect if a query is likely a follow-up to the previous conversation.

        Heuristics:
          - Short query (< 8 words) — likely a follow-up, not a new topic
          - Contains pronouns/references like "it", "that", "the same"
          - Starts with a conversational continuer like "what about", "and the"
        """
        if not self.history:
            return False

        query_lower = query.lower().strip()
        words = query_lower.split()

        # Short queries are almost always follow-ups when there's history
        if len(words) <= 5:
            return True

        # Check for follow-up signal words/phrases
        for signal in self.FOLLOW_UP_SIGNALS:
            if signal in query_lower:
                return True

        # Starts with a question word without a clear noun → follow-up
        # e.g., "What's the tolerance?" vs "What's the main bearing clearance?"
        if words[0] in ("what", "how", "why", "where", "when") and len(words) < 8:
            return True

        return False

    def expand_query_for_retrieval(self, query: str) -> str:
        """
        Expand a follow-up query with context from recent conversation
        to improve vector search retrieval.

        Strategy:
          1. If not a follow-up → return original query unchanged
          2. If follow-up → prepend key terms from last 1-2 exchanges
             so the embedding captures the broader topic context

        This does NOT use the LLM — it's a fast string operation.
        The expanded query is used ONLY for vector search, not shown to the user.
        """
        if not self.history or not self.is_follow_up(query):
            return query

        # Extract key context from recent exchanges
        context_terms = []

        # From last question — the topic being discussed
        last_q = self.history[-1]["question"]
        context_terms.append(last_q)

        # From last answer — extract key noun phrases (first 200 chars)
        # This captures specific technical terms the LLM mentioned
        last_a = self.history[-1].get("answer", "")[:200]
        if last_a:
            context_terms.append(last_a)

        # If there's a second-to-last exchange and query is very short,
        # include its question too for deeper context
        if len(self.history) >= 2 and len(query.split()) <= 4:
            prev_q = self.history[-2]["question"]
            context_terms.append(prev_q)

        # Build expanded query: original query + context
        # Put the user's actual query first so it has highest embedding weight
        context_snippet = " ".join(context_terms)
        # Truncate context to avoid oversized embedding input
        if len(context_snippet) > 500:
            context_snippet = context_snippet[:500]

        expanded = f"{query} [context: {context_snippet}]"

        logger.info(
            f"Query expanded for retrieval: '{query}' -> "
            f"'{expanded[:120]}...' (follow-up detected)"
        )
        return expanded

    def get_context_for_llm(self, max_exchanges: int = 3) -> str:
        """
        Build a structured conversation history block for the LLM prompt.

        Includes recent Q&A pairs so the LLM can:
          - Resolve pronouns ("it", "that", "the same")
          - Maintain diagnostic reasoning across turns
          - Avoid repeating information already given

        Returns empty string if no history.
        """
        if not self.history:
            return ""

        recent = self.history[-max_exchanges:]
        lines = [
            "## Previous Conversation",
            "(Use this context to understand follow-up questions. "
            "Do not repeat information already given unless asked.)",
            "",
        ]
        for i, exchange in enumerate(recent, 1):
            lines.append(f"**User [{i}]:** {exchange['question']}")
            # Truncate answers to keep prompt size reasonable
            answer_preview = exchange['answer'][:400]
            if len(exchange['answer']) > 400:
                answer_preview += "..."
            lines.append(f"**Assistant [{i}]:** {answer_preview}")

            # Include source references so LLM can avoid re-citing same sections
            if exchange.get("sources"):
                src_refs = []
                for s in exchange["sources"][:3]:  # max 3 refs per exchange
                    hierarchy = s.get("section_hierarchy", "") or s.get("section_title", "")
                    src_ref = s.get("source_file", "")
                    if hierarchy:
                        src_ref += f" > {hierarchy}"
                    src_ref += f" > Page {s.get('page_number', '?')}"
                    src_refs.append(src_ref)
                lines.append(f"*Sources used: {'; '.join(src_refs)}*")
            lines.append("")

        return "\n".join(lines)

    def get_context_summary(self) -> str:
        """Legacy method — returns brief summary for backward compatibility."""
        if not self.history:
            return ""
        lines = ["Previous conversation context:"]
        for i, exchange in enumerate(self.history[-3:], 1):
            lines.append(f"Q{i}: {exchange['question'][:200]}")
            lines.append(f"A{i}: {exchange['answer'][:300]}")
        return "\n".join(lines)

    def clear(self):
        self.history = []

    @property
    def count(self) -> int:
        return len(self.history)
