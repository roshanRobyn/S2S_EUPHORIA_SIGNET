"""
Small Language Model (SLM) wrapper for gloss-to-English translation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterable, List, Optional, Sequence

try:
    # Added AutoConfig to the imports
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoConfig = None
    torch = None

try:
    import ollama
except Exception:
    ollama = None


DEFAULT_BASE_MODEL_NAME = "microsoft/phi-2"
DEFAULT_FINETUNED_PATH = "checkpoints/slm-aslgpc12"

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
    gloss_str = " ".join(gloss_tokens)
    nmm_str = " ".join(nmm_tokens) if nmm_tokens else ""
    if nmm_str:
        return f"Gloss: {gloss_str}\nNMM: {nmm_str}\nEnglish:"
    return f"Gloss: {gloss_str}\nEnglish:"


def _split_on_whitespace(text: str) -> List[str]:
    return [t for t in text.strip().split() if t]


@dataclass
class SLMConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    use_ollama: bool = False
    ollama_model: Optional[str] = None


class SLMClient:
    def __init__(self, config: Optional[SLMConfig] = None) -> None:
        self.config = config or SLMConfig()

        self._tokenizer = None
        self._model = None

        if not self.config.use_ollama:
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError(
                    "transformers is not installed; install `transformers[torch]`"
                )

            # --- 🛠️ THE FIX FOR PHI-2 ---
            # 1. Load the tokenizer first
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, 
                trust_remote_code=True
            )
            
            # 2. Load the Config separately
            config_obj = AutoConfig.from_pretrained(
                self.config.model_name, 
                trust_remote_code=True
            )
            
            # 3. MANUALLY PATCH THE MISSING ATTRIBUTE
            # Phi-2 often lacks a pad_token_id, causing the crash.
            # We set it to the eos_token_id (End of String) to stop the error.
            if getattr(config_obj, "pad_token_id", None) is None:
                config_obj.pad_token_id = self._tokenizer.eos_token_id

            # 4. Load the Model with the Patched Config
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                config=config_obj,  # Pass our fixed config here
                trust_remote_code=True
            )

            if hasattr(self._model, "to") and torch is not None:
                self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()

        else:
            if ollama is None:
                raise RuntimeError("ollama Python client is not installed.")
            if not self.config.ollama_model:
                raise ValueError("When use_ollama=True, `ollama_model` must be set.")

    async def stream_translate(
        self,
        gloss_tokens: Sequence[str],
        nmm_tokens: Sequence[str],
    ) -> AsyncIterator[str]:
        user_prompt = _format_prompt(gloss_tokens, nmm_tokens)

        if self.config.use_ollama:
            async for token in self._ollama_stream_chat(SYSTEM_PROMPT, user_prompt):
                for word in _split_on_whitespace(token):
                    yield word
        else:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._hf_generate, SYSTEM_PROMPT, user_prompt
            )
            for word in _split_on_whitespace(text):
                yield word

    def _hf_generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Hugging Face model/tokenizer not initialized.")

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=50, # Reduced to prevent rambling
                do_sample=True,
                temperature=0.6,   # Lower temp = less creative/hallucinating
                top_p=0.90,
                pad_token_id=self._tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        decoded_text = self._tokenizer.decode(generated, skip_special_tokens=True)
        
        # --- ✂️ THE FIX: CUT OFF HALLUCINATIONS ---
        # Stop at the first newline or "Gloss:" marker
        if "\n" in decoded_text:
            decoded_text = decoded_text.split("\n")[0]
        if "Gloss:" in decoded_text:
            decoded_text = decoded_text.split("Gloss:")[0]
            
        return decoded_text.strip()
    async def _ollama_stream_chat(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        if ollama is None:
            raise RuntimeError("Ollama client not available.")

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
        else:
            loop = asyncio.get_event_loop()
            def _sync_stream():
                for part in ollama.chat(
                    model=self.config.ollama_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                ):
                    yield part.get("message", {}).get("content", "")
            for token in await loop.run_in_executor(None, lambda: list(_sync_stream())):
                if token:
                    yield token