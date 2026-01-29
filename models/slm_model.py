"""
Small Language Model (SLM) wrapper for gloss-to-English translation.

This module encapsulates the "Brain" of the system. It is responsible
for taking normalized ISL gloss tokens and non-manual marker (NMM)
tokens and producing natural English text, using either a local
Transformers model (e.g., Phi-3 Mini, Gemma-2B, T5) or a remote Ollama
deployment with streaming support.

The default implementation uses Hugging Face `transformers` with a
causal language model. You can swap in your preferred model by changing
the MODEL_NAME and generation parameters below.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterable, List, Optional, Sequence

try:  # optional, only needed if you use Hugging Face models
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type import
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

try:  # optional, only if you use Ollama
    import ollama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ollama = None  # type: ignore


DEFAULT_BASE_MODEL_NAME = "microsoft/phi-2"
DEFAULT_FINETUNED_PATH = "checkpoints/slm-aslgpc12"

# Prefer a locally fine-tuned checkpoint if it exists; otherwise fall back
# to the base model name. This way you can run inference immediately after
# training without changing other code, but things still work before
# training has completed.
DEFAULT_MODEL_NAME = (
    DEFAULT_FINETUNED_PATH
    if Path(DEFAULT_FINETUNED_PATH).exists()
    else DEFAULT_BASE_MODEL_NAME
)

SYSTEM_PROMPT = (
    "System: Translate these ISL glosses into natural English. "
    "Use NMM tokens for tone and punctuation."
)


def _format_prompt(gloss_tokens: Sequence[str], nmm_tokens: Sequence[str]) -> str:
    """
    Build the user prompt from gloss and NMM tokens.

    Example:
        gloss_tokens = ["MY", "BROTHER", "DOCTOR"]
        nmm_tokens   = ["[QUESTION]"]
        ->
        'Gloss: MY BROTHER DOCTOR\nNMM: [QUESTION]\nEnglish:'
    """
    gloss_str = " ".join(gloss_tokens)
    nmm_str = " ".join(nmm_tokens) if nmm_tokens else ""
    if nmm_str:
        return f"Gloss: {gloss_str}\nNMM: {nmm_str}\nEnglish:"
    return f"Gloss: {gloss_str}\nEnglish:"


def _split_on_whitespace(text: str) -> List[str]:
    # Simple word-level splitting for streaming.
    return [t for t in text.strip().split() if t]


@dataclass
class SLMConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    use_ollama: bool = False
    ollama_model: Optional[str] = None  # e.g. "phi3:mini"


class SLMClient:
    """
    High-level client that provides a single async `stream_translate` API.

    You can back this client by:
    - A local Transformers model (default), or
    - An Ollama server with streaming enabled.
    """

    def __init__(self, config: Optional[SLMConfig] = None) -> None:
        self.config = config or SLMConfig()

        self._tokenizer = None
        self._model = None

        if not self.config.use_ollama:
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError(
                    "transformers is not installed; install `transformers[torch]` "
                    "or set `use_ollama=True` in SLMConfig."
                )

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            if hasattr(self._model, "to") and torch is not None:
                self._model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
            self._model.eval()

        else:
            if ollama is None:
                raise RuntimeError(
                    "ollama Python client is not installed; run `pip install ollama` "
                    "or disable `use_ollama`."
                )
            if not self.config.ollama_model:
                raise ValueError("When use_ollama=True, `ollama_model` must be set.")

    async def stream_translate(
        self,
        gloss_tokens: Sequence[str],
        nmm_tokens: Sequence[str],
    ) -> AsyncIterator[str]:
        """
        High-level translation API.

        Args:
            gloss_tokens: Normalized ISL gloss tokens.
            nmm_tokens: Non-manual marker tokens, e.g. ["[QUESTION]", "[HAPPY]"].

        Yields:
            Individual word-level strings suitable for streaming into TTS.
        """
        user_prompt = _format_prompt(gloss_tokens, nmm_tokens)

        if self.config.use_ollama:
            # Delegate to Ollama's streaming API.
            async for token in self._ollama_stream_chat(SYSTEM_PROMPT, user_prompt):
                for word in _split_on_whitespace(token):
                    yield word
        else:
            # Use local Transformers model; emulate streaming by chunking the
            # full generation into word-level tokens.
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._hf_generate, SYSTEM_PROMPT, user_prompt
            )
            for word in _split_on_whitespace(text):
                yield word

    # ------------------------------------------------------------------
    # Hugging Face backend
    # ------------------------------------------------------------------

    def _hf_generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Hugging Face model/tokenizer not initialized.")

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
        )
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[attr-defined]

        with torch.no_grad():  # type: ignore[union-attr]
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    async def _ollama_stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from an Ollama model, assuming `ollama` Python client.

        The model should be loaded and available via `ollama pull <model>`.
        """
        if ollama is None:
            raise RuntimeError("Ollama client not available.")

        # ollama.AsyncClient is available in recent versions; fall back to sync if needed.
        if hasattr(ollama, "AsyncClient"):
            client = ollama.AsyncClient()

            async for part in client.chat(
                model=self.config.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            ):
                content = part.get("message", {}).get("content", "")
                if content:
                    yield content
        else:  # pragma: no cover - legacy / sync fallback
            loop = asyncio.get_event_loop()

            def _sync_stream():
                for part in ollama.chat(  # type: ignore[call-arg]
                    model=self.config.ollama_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                ):
                    yield part.get("message", {}).get("content", "")

            # Bridge sync generator into async iterator
            for token in await loop.run_in_executor(None, lambda: list(_sync_stream())):
                if token:
                    yield token


__all__ = [
    "SLMConfig",
    "SLMClient",
    "SYSTEM_PROMPT",
]

