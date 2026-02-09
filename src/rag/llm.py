"""LLM client for RAG: build prompt and call API (OpenAI or callable)."""
from __future__ import annotations

from typing import Callable

from src.schema import Chunk

DEFAULT_SYSTEM = "Answer based only on the context. If the context does not contain the answer, say so briefly."


def build_context(chunks: list[Chunk], max_chars: int = 4000) -> str:
    """Concatenate chunk texts up to max_chars."""
    out = []
    n = 0
    for c in chunks:
        if n + len(c.text) + 2 > max_chars:
            break
        out.append(c.text)
        n += len(c.text) + 2
    return "\n\n".join(out)


def build_prompt(query: str, context: str, system: str = DEFAULT_SYSTEM) -> str:
    """Single user message for chat API."""
    return f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def openai_generate(query: str, context: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256) -> str:
    """Call OpenAI Chat Completions. Requires OPENAI_API_KEY."""
    import os
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = build_prompt(query, context)
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def default_llm_client() -> Callable[[str, str], str] | None:
    """Return openai_generate if OPENAI_API_KEY set, else None."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    # Ensure .env is loaded from project root (in case config.settings wasn't imported yet)
    _root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(dotenv_path=str(_root / ".env"), encoding="utf-8")
    if (os.getenv("OPENAI_API_KEY") or "").strip():
        return openai_generate
    return None
