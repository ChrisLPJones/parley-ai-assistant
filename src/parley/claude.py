"""Claude via the Anthropic API (backend = "anthropic"), with Parley's local tools plus Anthropic's
server-side web search and web fetch."""
import logging
import os
from typing import Callable, Iterator

import anthropic

from . import credstore
from .config import LlmConfig
from .tools import Toolbox

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 6
WEB_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3},
]


class ClaudeProvider:
    def __init__(self, cfg: LlmConfig, tb: Toolbox | None, web: bool):
        self.cfg, self.toolbox = cfg, tb
        self.model = cfg.model or DEFAULT_MODEL
        key = credstore.resolve_api_key("anthropic", cfg)
        if not key:
            raise SystemExit('backend = "anthropic" needs an API key: run "parley --set-key anthropic" or set ANTHROPIC_API_KEY')
        if os.environ.get("ANTHROPIC_BASE_URL"):  # the SDK honours it, so the key would go there
            credstore.warn_if_custom_endpoint("anthropic", os.environ["ANTHROPIC_BASE_URL"])
        self.client = anthropic.Anthropic(api_key=key)
        self.tools = (tb.anthropic() if tb else []) + (WEB_TOOLS if web else [])

    def describe(self) -> str:
        return f"Claude {self.model} (effort {self.cfg.effort})"

    def warm_up(self) -> None:
        # Fails fast on a bad key or model name instead of on the first question
        try:
            self.client.models.retrieve(self.model)
        except anthropic.AuthenticationError as e:
            raise SystemExit("Anthropic rejected the API key (replace it with 'parley --set-key anthropic')") from e
        except anthropic.NotFoundError as e:
            raise SystemExit(f"Claude model {self.model!r} not found. Set llm.model in config.toml.") from e

    def run(self, system: str, history: list[dict], cancelled: Callable[[], bool]) -> Iterator[str]:
        json_retries = 0
        spoke_before = False  # anything said earlier in this reply
        for _ in range(MAX_TOOL_ROUNDS):
            spoke = False
            try:
                # Leaving the `with` block (including when Chat closes this generator on an
                # interruption) closes the connection, so Claude stops generating.
                with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=64000,
                    system=system,
                    messages=history,
                    tools=self.tools,
                    # Spoken chat doesn't need deep reasoning; low effort keeps time-to-first-word short
                    output_config={"effort": self.cfg.effort},
                    # If a safety classifier declines, the API retries on Anthropic's recommended fallback model
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    for event in stream:
                        if cancelled():
                            return
                        if event.type == "text":
                            spoke = True
                            yield event.text
                    response = stream.get_final_message()
                json_retries = 0
            except ValueError:
                # Tool input JSON the SDK couldn't parse at all; re-issue the turn (bounded).
                # API errors aren't ValueError and propagate.
                json_retries += 1
                if json_retries > 2:
                    raise
                continue

            history.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":  # server-side web search wants to keep going
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "refusal":
                if not spoke:
                    yield "Sorry, I can't help with that one."
                return
            if not tool_uses:
                return
            if response.stop_reason == "max_tokens":
                raise RuntimeError("Claude's tool request was cut off")  # never run a truncated input
            # Speak "Let me check" now, before the tools run, or a filler if they're slow
            yield "\n\n" if spoke else self.toolbox.filler([b.name for b in tool_uses], spoke_before)
            spoke_before = True
            results = []
            for block in tool_uses:
                # Toolbox.call validates the input before running anything (eager streaming
                # means the API no longer does)
                text, error = self.toolbox.call(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text,
                                **({"is_error": True} if error else {})})
            history.append({"role": "user", "content": results})
        yield "Sorry, that took too many steps, so I stopped."
