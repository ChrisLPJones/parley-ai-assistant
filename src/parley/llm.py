"""Streaming chat. Conversation history lives in Chat; each backend is a small provider that streams
one reply: Parley's built-in llama-server, Ollama, any OpenAI-compatible server, OpenAI, or Claude."""
import json
import logging
import os
import time
from typing import Callable, Iterator

import httpx

from .config import LlmConfig

log = logging.getLogger(__name__)

CLOUD_BACKENDS = {"openai": "OpenAI", "anthropic": "Anthropic"}


class Chat:
    def __init__(self, cfg: LlmConfig, provider):
        self.cfg = cfg
        self.provider = provider
        self.history: list[dict] = []
        self.last_used = 0.0

    def warm_up(self) -> None:
        start = time.perf_counter()
        self.provider.warm_up()
        log.info("LLM ready: %s (%.1fs)", self.provider.describe(), time.perf_counter() - start)

    def reply(self, text: str, cancelled: Callable[[], bool]) -> Iterator[str]:
        """Yield the reply in pieces as they're generated; stops early if cancelled() turns true."""
        if self.history and time.monotonic() - self.last_used > self.cfg.reset_after_idle_s:
            log.info("Idle for a while, starting a fresh conversation")
            self.history = []
        self.history = self.history[-2 * self.cfg.history_turns:] + [{"role": "user", "content": text}]
        stream = self.provider.stream(self.cfg.system_prompt, self.history)
        answer = []
        try:
            for piece in stream:
                if cancelled():
                    break
                answer.append(piece)
                yield piece
        finally:
            stream.close()  # closes the HTTP response, so the server stops generating
            self.last_used = time.monotonic()
            if answer:
                self.history.append({"role": "assistant", "content": "".join(answer)})
            else:
                self.history.pop()  # keep user/assistant turns paired


def create(cfg: LlmConfig, base_url: str | None = None, local_model: str = "local") -> Chat:
    """Build the chat for cfg.backend. base_url/local_model describe the built-in llama-server."""
    if cfg.backend == "anthropic":
        from .claude import ClaudeProvider
        return Chat(cfg, ClaudeProvider(cfg))
    if cfg.backend == "ollama":
        return Chat(cfg, OllamaProvider(cfg, cfg.base_url or "http://127.0.0.1:11434"))
    if cfg.backend == "openai":
        key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit('backend = "openai" needs an API key: set OPENAI_API_KEY or llm.api_key')
        return Chat(cfg, OpenAICompatibleProvider(
            cfg, cfg.base_url or "https://api.openai.com/v1", key, cfg.model or "chat-latest", cloud=True))
    if cfg.backend == "server":
        return Chat(cfg, OpenAICompatibleProvider(cfg, cfg.base_url or "http://127.0.0.1:8080/v1", cfg.api_key, cfg.model))
    if cfg.backend == "builtin":
        return Chat(cfg, OpenAICompatibleProvider(cfg, base_url, None, local_model, builtin=True))
    raise SystemExit(f"Unknown llm.backend {cfg.backend!r}; use builtin, anthropic, openai, ollama or server")


class OpenAICompatibleProvider:
    """Chat Completions streaming: OpenAI itself, llama-server, LM Studio, vLLM, ..."""

    def __init__(self, cfg: LlmConfig, base_url: str, api_key: str | None, model: str,
                 cloud: bool = False, builtin: bool = False):
        self.cfg, self.cloud, self.builtin = cfg, cloud, builtin
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(15, read=300))
        self.model = model or self._first_model()

    def _first_model(self) -> str:
        names = [m["id"] for m in self._check(self.client.get("/models")).json()["data"]]
        if not names:
            raise RuntimeError(f"No models available on {self.client.base_url}")
        log.info("No llm.model set, using %s", names[0])
        return names[0]

    def describe(self) -> str:
        return f"OpenAI {self.model}" if self.cloud else self.model

    def warm_up(self) -> None:
        if self.cloud:
            # Checks the key and model name up front, without paying for a completion
            self._check(self.client.get(f"/models/{self.model}"))
            return
        # Makes a local server load the model now rather than on the first question
        self._check(self.client.post("/chat/completions", json=self._body(
            [{"role": "user", "content": "hi"}], max_tokens=1, stream=False)))

    def _body(self, messages: list[dict], max_tokens: int | None = None, stream: bool = True) -> dict:
        body: dict = {"model": self.model, "messages": messages, "stream": stream}
        if not self.cloud:
            body["temperature"] = self.cfg.temperature  # current OpenAI models only accept the default
        if max_tokens:
            body["max_tokens"] = max_tokens
        if self.builtin:
            body["chat_template_kwargs"] = {"enable_thinking": self.cfg.think}
        return body

    def _check(self, r: httpx.Response) -> httpx.Response:
        if r.status_code < 400:
            return r
        r.read()
        try:
            detail = r.json().get("error", {}).get("message") or r.text
        except (ValueError, AttributeError):
            detail = r.text
        if r.status_code == 401:
            raise SystemExit(f"{'OpenAI' if self.cloud else 'The server'} rejected the API key: {detail}")
        if r.status_code == 404 and self.cloud:
            raise SystemExit(f"OpenAI model {self.model!r} isn't available ({detail}). Set llm.model in config.toml.")
        raise RuntimeError(f"HTTP {r.status_code} from {self.client.base_url}: {detail}")

    def stream(self, system: str, history: list[dict]) -> Iterator[str]:
        messages = [{"role": "system", "content": system}] + history
        with self.client.stream("POST", "/chat/completions", json=self._body(messages)) as r:
            self._check(r)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                choices = json.loads(payload).get("choices") or [{}]
                if piece := choices[0].get("delta", {}).get("content"):
                    yield piece


class OllamaProvider:
    def __init__(self, cfg: LlmConfig, base_url: str):
        self.cfg = cfg
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(10, read=300))
        self.model = cfg.model or self._first_model()

    def _first_model(self) -> str:
        names = [m["name"] for m in self.client.get("/api/tags").raise_for_status().json()["models"]]
        if not names:
            raise RuntimeError(f"No models installed in Ollama at {self.client.base_url}")
        log.info("No llm.model set, using %s", names[0])
        return names[0]

    def describe(self) -> str:
        return f"Ollama {self.model}"

    def _options(self) -> dict:
        options = {"temperature": self.cfg.temperature}
        if self.cfg.num_ctx:
            options["num_ctx"] = self.cfg.num_ctx  # a different value than the loaded model's forces a reload
        return options

    def warm_up(self) -> None:
        self.client.post("/api/generate", json={
            "model": self.model, "keep_alive": self.cfg.keep_alive, "options": self._options(),
        }).raise_for_status()

    def stream(self, system: str, history: list[dict]) -> Iterator[str]:
        body = {
            "model": self.model, "messages": [{"role": "system", "content": system}] + history, "stream": True,
            "think": self.cfg.think, "keep_alive": self.cfg.keep_alive, "options": self._options(),
        }
        with self.client.stream("POST", "/api/chat", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if piece := data.get("message", {}).get("content"):
                    yield piece
                if data.get("done"):
                    return
