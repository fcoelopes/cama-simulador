"""Abstração simples sobre os LLMs suportados."""
from __future__ import annotations
from typing import Protocol

from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.models import list_models


class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite-preview"):
        configure(api_key=api_key)
        self._model_name = model
        self._model = GenerativeModel(model_name=model)

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:    
        model_inst = self._model
        if system:
            model_inst = GenerativeModel(
                model_name=self._model_name,
                system_instruction=system
            )

        resp = model_inst.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 8192},
        )
        
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.text
        
        return ""


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=temperature,
            max_tokens=8192,
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # concatena blocos de texto
        text_parts = []
        for block in resp.content:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                text_parts.append(block_text)
        return "".join(text_parts)
    
class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        from openai import OpenAI
        # DeepSeek usa o SDK da OpenAI, mas com base_url diferente
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# class OllamaClient:
#     def __init__(self, api_key: str = "ollama", model: str = "deepseek-r1:14b"):
#         from openai import OpenAI
#         # Ollama local geralmente roda na porta 11434
#         self._client = OpenAI(api_key=api_key, base_url="http://localhost:11434/v1")
#         self._model = model

#     def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
#         msgs = []
#         if system:
#             msgs.append({"role": "system", "content": system})
#         msgs.append({"role": "user", "content": prompt})
        
#         try:
#             resp = self._client.chat.completions.create(
#                 model=self._model,
#                 messages=msgs,
#                 temperature=temperature,
#             )
#             return resp.choices[0].message.content or ""
#         except Exception as e:
#             return f"Erro ao conectar ao Ollama local: {e}"


PROVIDERS = {
    "Gemini": {
        "class": GeminiClient,
        "models": ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite", "gemini-3-pro-preview"],
    },
    "OpenAI": {
        "class": OpenAIClient,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "Anthropic": {
        "class": AnthropicClient,
        "models": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
    },
    "DeepSeek": {
        "class": DeepSeekClient,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    # "Ollama": {
    #     "class": OllamaClient,
    #     "models": ["deepseek-r1:14b", "gemma-3-27b-it", "llama-4-8b"],
    # },
}


def build_client(provider: str, api_key: str, model: str) -> LLMClient:
    cfg = PROVIDERS[provider]
    return cfg["class"](api_key=api_key, model=model)