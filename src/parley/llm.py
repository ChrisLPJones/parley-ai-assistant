"""Streaming chat with tool use. Chat owns the conversation; each backend is a provider whose run()
streams one reply, running any tools the model asks for along the way and recording the whole
exchange in the history: Parley's built-in llama-server, Ollama, any OpenAI-compatible server,
OpenAI, or Claude."""
import datetime
import json
import logging
import time
from typing import Callable, Iterator

import httpx

from . import credstore
from .config import LlmConfig, ToolsConfig
from .tools import TOOLS_PROMPT, Toolbox, toolbox

log = logging.getLogger(__name__)

CLOUD_BACKENDS = {"openai": "OpenAI", "anthropic": "Anthropic"}
MAX_TOOL_ROUNDS = 6
TOO_MANY_STEPS = "Sorry, that took too many steps, so I stopped."


class Chat:
    def __init__(self, cfg: LlmConfig, provider, web: bool):
        self.cfg = cfg
        self.provider = provider
        self.web = web
        self.history: list[dict] = []
        self.last_used = 0.0

    def rebind(self, cfg: LlmConfig) -> None:
        """Read live settings from `cfg` from now on (the app's shared LlmConfig, after a reload built this chat from a copy)."""
        self.cfg = cfg
        self.provider.cfg = cfg

    def warm_up(self) -> None:
        start = time.perf_counter()
        self.provider.warm_up()
        log.info("LLM ready: %s (%.1fs)", self.provider.describe(), time.perf_counter() - start)

    def _system(self) -> str:
        prompt = self.cfg.system_prompt + f" Today is {datetime.date.today():%A %d %B %Y}."
        if self.provider.toolbox:
            prompt += TOOLS_PROMPT.format(web=", and search the web" if self.web else "")
        return prompt

    def reply(self, text: str, cancelled: Callable[[], bool]) -> Iterator[str]:
        """Yield the spoken reply in pieces as it's generated; stops early if cancelled() turns true."""
        if self.history and time.monotonic() - self.last_used > self.cfg.reset_after_idle_s:
            log.info("Idle for a while, starting a fresh conversation")
            self.history = []
        start = len(self.history)
        self.history.append({"role": "user", "content": text})
        done = spoke = False
        try:
            for piece in self.provider.run(self._system(), self.history, cancelled):
                spoke = spoke or bool(piece.strip())
                yield piece
            done = not cancelled()
            if done and not spoke:
                yield "Sorry, I didn't come up with an answer for that."
        finally:
            self.last_used = time.monotonic()
            if done:
                self._trim()
            else:
                # Interrupted or failed: drop the half-finished exchange (it may end in a tool call
                # with no result, which the next request would be rejected for)
                del self.history[start:]

    def _trim(self) -> None:
        # Cut at a user question, never between a tool call and its result
        questions = [i for i, m in enumerate(self.history) if m["role"] == "user" and isinstance(m["content"], str)]
        if len(questions) > self.cfg.history_turns:
            del self.history[:questions[-self.cfg.history_turns]]


def create(cfg: LlmConfig, tools_cfg: ToolsConfig, base_url: str | None = None, local_model: str = "local") -> Chat:
    """Build the chat for cfg.backend. base_url/local_model describe the built-in llama-server."""
    web = tools_cfg.enabled and tools_cfg.web_search
    if cfg.backend == "anthropic":
        from .claude import ClaudeProvider
        # Claude brings its own (better) web search and fetch, so the local web tools aren't added
        tb = toolbox(web, local_web=False) if tools_cfg.enabled else None
        return Chat(cfg, ClaudeProvider(cfg, tb, web), web)
    tb = toolbox(web) if tools_cfg.enabled else None
    if cfg.backend == "ollama":
        return Chat(cfg, OllamaProvider(cfg, cfg.base_url or "http://127.0.0.1:11434", tb), web)
    if cfg.backend == "openai":
        key = credstore.resolve_api_key("openai", cfg)
        if not key:
            raise SystemExit('backend = "openai" needs an API key: run "parley --set-key openai" or set OPENAI_API_KEY')
        credstore.warn_if_custom_endpoint("openai", cfg.base_url or "https://api.openai.com/v1")
        return Chat(cfg, OpenAICompatibleProvider(
            cfg, tb, cfg.base_url or "https://api.openai.com/v1", key, cfg.model or "chat-latest", cloud=True), web)
    if cfg.backend == "server":
        url = cfg.base_url or "http://127.0.0.1:8080/v1"
        if cfg.api_key:  # a server key stays in config.toml (it is not an Anthropic / OpenAI key)
            credstore.register_secret(cfg.api_key)
            credstore.warn_if_custom_endpoint("server", url, loopback_ok=True)
        return Chat(cfg, OpenAICompatibleProvider(cfg, tb, url, cfg.api_key, cfg.model), web)
    if cfg.backend == "builtin":
        return Chat(cfg, OpenAICompatibleProvider(cfg, tb, base_url, None, local_model, builtin=True), web)
    raise SystemExit(f"Unknown llm.backend {cfg.backend!r}; use builtin, anthropic, openai, ollama or server")


class OpenAICompatibleProvider:
    """Chat Completions streaming with function calling: OpenAI itself, llama-server, LM Studio, vLLM, ..."""

    def __init__(self, cfg: LlmConfig, tb: Toolbox | None, base_url: str, api_key: str | None, model: str,
                 cloud: bool = False, builtin: bool = False):
        self.cfg, self.toolbox, self.cloud, self.builtin = cfg, tb, cloud, builtin
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
        if r.status_code == 401:  # the provider's detail text can quote part of the key, so it is not repeated
            raise SystemExit(f"{'OpenAI' if self.cloud else 'The server'} rejected the API key.")
        if r.status_code == 404 and self.cloud:
            raise SystemExit(f"OpenAI model {self.model!r} isn't available ({detail}). Set llm.model in config.toml.")
        raise RuntimeError(f"HTTP {r.status_code} from {self.client.base_url}: {detail}")

    def run(self, system: str, history: list[dict], cancelled: Callable[[], bool]) -> Iterator[str]:
        spoke = False
        for _ in range(MAX_TOOL_ROUNDS):
            body = self._body([{"role": "system", "content": system}] + history)
            if self.toolbox:
                body["tools"] = self.toolbox.openai()
            text, calls = [], {}
            with self.client.stream("POST", "/chat/completions", json=body) as r:
                self._check(r)
                for line in r.iter_lines():
                    if cancelled():
                        return
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta") or {}
                    if piece := delta.get("content"):
                        text.append(piece)
                        yield piece
                    for tc in delta.get("tool_calls") or []:  # arrives in fragments, keyed by index
                        call = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        call["id"] = tc.get("id") or call["id"]
                        call["name"] += fn.get("name") or ""
                        call["arguments"] += fn.get("arguments") or ""
            message = {"role": "assistant", "content": "".join(text)}
            if not calls:
                history.append(message)
                return
            message["tool_calls"] = [
                {"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for i, c in sorted(calls.items())
            ]
            history.append(message)
            # Speak "Let me check" now, before the tools run, or a filler if they're slow
            yield "\n\n" if text else self.toolbox.filler([tc["function"]["name"] for tc in message["tool_calls"]], spoke)
            spoke = True
            for tc in message["tool_calls"]:
                result, _ = self.toolbox.call(tc["function"]["name"], tc["function"]["arguments"])
                history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        yield TOO_MANY_STEPS


class OllamaProvider:
    def __init__(self, cfg: LlmConfig, base_url: str, tb: Toolbox | None):
        self.cfg, self.toolbox = cfg, tb
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

    def run(self, system: str, history: list[dict], cancelled: Callable[[], bool]) -> Iterator[str]:
        spoke = False
        for _ in range(MAX_TOOL_ROUNDS):
            body = {
                "model": self.model, "messages": [{"role": "system", "content": system}] + history, "stream": True,
                "think": self.cfg.think, "keep_alive": self.cfg.keep_alive, "options": self._options(),
            }
            if self.toolbox:
                body["tools"] = self.toolbox.openai()
            text, calls = [], []
            with self.client.stream("POST", "/api/chat", json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if cancelled():
                        return
                    if not line:
                        continue
                    message = json.loads(line).get("message", {})
                    if piece := message.get("content"):
                        text.append(piece)
                        yield piece
                    calls += message.get("tool_calls") or []
            history.append({"role": "assistant", "content": "".join(text), **({"tool_calls": calls} if calls else {})})
            if not calls:
                return
            yield "\n\n" if text else self.toolbox.filler([c.get("function", {}).get("name", "") for c in calls], spoke)
            spoke = True
            for call in calls:
                fn = call.get("function", {})
                result, _ = self.toolbox.call(fn.get("name", ""), fn.get("arguments") or {})
                history.append({"role": "tool", "content": result, "tool_name": fn.get("name", "")})
        yield TOO_MANY_STEPS
