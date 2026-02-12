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
from typing import Optional, Generator

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.3:8b")


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
    """
    if not retrieved_chunks:
        return "No relevant manual data found for this query."

    context_parts = []
    context_parts.append("=" * 60)
    context_parts.append("RETRIEVED MANUAL DATA (use ONLY this data to answer)")
    context_parts.append("=" * 60)

    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source_file", "Unknown")
        page = chunk.get("page_number", "?")
        ctype = chunk.get("chunk_type", "text")
        distance = chunk.get("distance", 0)
        relevance = max(0, round((1 - distance) * 100, 1))

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
            f"[Relevance: {relevance}%] ---"
        )
        context_parts.append(chunk.get("text", ""))

    context_parts.append("\n" + "=" * 60)
    context_parts.append("CITATION INSTRUCTION: When referencing this data, cite as:")
    context_parts.append("**[Source: filename > section hierarchy > Page X]**")
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


def generate_response(
    question: str,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    equipment_name: str = "",
    stream: bool = True,
) -> Generator[str, None, None]:
    """
    Generate a diagnostic response using the local LLM — STREAMING.
    Yields response chunks as they arrive.
    """
    import ollama

    context = build_context(retrieved_chunks)

    equipment_context = ""
    if equipment_name:
        equipment_context = f"\nYou are answering questions about: **{equipment_name}**\n"

    user_message = f"""{equipment_context}
## User Question
{question}

## Manual Data
{context}

Provide your analysis based STRICTLY on the manual data above. Cite every key claim with **[Source: filename > section > Page X]**. If the data doesn't contain relevant information, clearly state that."""

    client = ollama.Client(host=OLLAMA_BASE_URL)

    logger.info(f"Sending to Ollama model '{model}':")
    logger.info(f"System Prompt:\n{SYSTEM_PROMPT}")
    logger.info(f"User Message:\n{user_message}")

    response_stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    for chunk in response_stream:
        logger.debug(f"Raw chunk: {chunk}") # New debug line
        if isinstance(chunk, dict) and "message" in chunk and "content" in chunk["message"]:
            logger.debug(f"Received chunk: {chunk['message']['content']}")
            yield chunk["message"]["content"]
        elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
            logger.debug(f"Received chunk (legacy): {chunk.message.content}")
            yield chunk.message.content
        else:
            logger.warning(f"Unexpected chunk format: {chunk}")



def generate_response_full(
    question: str,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    equipment_name: str = "",
) -> str:
    """Non-streaming version: returns the complete response as a string."""
    import ollama

    context = build_context(retrieved_chunks)

    equipment_context = ""
    if equipment_name:
        equipment_context = f"\nYou are answering questions about: **{equipment_name}**\n"

    user_message = f"""{equipment_context}
## User Question
{question}

## Manual Data
{context}

Provide your analysis based STRICTLY on the manual data above. Cite every key claim with **[Source: filename > section > Page X]**."""

    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=False,
    )

    if isinstance(response, dict):
        return response.get("message", {}).get("content", "No response generated.")
    return str(response.message.content) if hasattr(response, 'message') else "No response generated."


# ---------------------------------------------------------------------------
# Conversation memory (per-session)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Maintains conversation context for follow-up questions.
    Keeps the last N exchanges for multi-turn diagnostic conversations.
    """

    def __init__(self, max_history: int = 10):
        self.history: list[dict] = []
        self.max_history = max_history

    def add_exchange(self, question: str, answer: str, sources: list[dict] = None):
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources or [],
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_summary(self) -> str:
        """Get a summary of recent conversation for context."""
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
