"""Streaming chat against a local LLM server (Ollama native API or any OpenAI-compatible server)."""
import json
import logging
import time
from typing import Callable, Iterator

import httpx

from .config import LlmConfig

log = logging.getLogger(__name__)


class Chat:
    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self.client = httpx.Client(base_url=cfg.base_url.rstrip("/"), timeout=httpx.Timeout(10, read=300))
        self.model = cfg.model or self._first_model()
        self.history: list[dict] = []
        self.last_used = 0.0

    def _first_model(self) -> str:
        if self.cfg.api == "ollama":
            names = [m["name"] for m in self.client.get("/api/tags").raise_for_status().json()["models"]]
        else:
            names = [m["id"] for m in self.client.get("/models").raise_for_status().json()["data"]]
        if not names:
            raise RuntimeError(f"No models available on {self.cfg.base_url}")
        log.info("No llm.model set, using %s", names[0])
        return names[0]

    def _ollama_options(self) -> dict:
        options = {"temperature": self.cfg.temperature}
        if self.cfg.num_ctx:
            options["num_ctx"] = self.cfg.num_ctx  # a different value than the loaded model's forces a reload
        return options

    def warm_up(self) -> None:
        """Load the model into VRAM now rather than on the first question."""
        start = time.perf_counter()
        if self.cfg.api == "ollama":
            self.client.post("/api/generate", json={
                "model": self.model, "keep_alive": self.cfg.keep_alive, "options": self._ollama_options(),
            }).raise_for_status()
        else:
            list(self._stream_openai([{"role": "user", "content": "hi"}], max_tokens=1))
        log.info("LLM ready: %s (%.1fs)", self.model, time.perf_counter() - start)

    def reply(self, text: str, cancelled: Callable[[], bool]) -> Iterator[str]:
        """Yield the reply in pieces as they're generated; stops early if cancelled() turns true."""
        if self.history and time.monotonic() - self.last_used > self.cfg.reset_after_idle_s:
            log.info("Idle for a while, starting a fresh conversation")
            self.history = []
        self.history = self.history[-2 * self.cfg.history_turns:] + [{"role": "user", "content": text}]
        messages = [{"role": "system", "content": self.cfg.system_prompt}] + self.history
        stream = self._stream_ollama(messages) if self.cfg.api == "ollama" else self._stream_openai(messages)
        answer = []
        try:
            for piece in stream:
                if cancelled():
                    break
                answer.append(piece)
                yield piece
        finally:
            stream.close()
            self.last_used = time.monotonic()
            if answer:
                self.history.append({"role": "assistant", "content": "".join(answer)})
            else:
                self.history.pop()  # keep user/assistant turns paired

    def _stream_ollama(self, messages: list[dict]) -> Iterator[str]:
        body = {
            "model": self.model, "messages": messages, "stream": True, "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive, "options": self._ollama_options(),
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

    def _stream_openai(self, messages: list[dict], **extra) -> Iterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True, "temperature": self.cfg.temperature, **extra}
        with self.client.stream("POST", "/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                choices = json.loads(payload).get("choices") or [{}]
                if piece := choices[0].get("delta", {}).get("content"):
                    yield piece
