"""Claude via the Anthropic API (backend = "anthropic")."""
import logging
import os
from typing import Iterator

import anthropic

from .config import LlmConfig

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeProvider:
    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self.model = cfg.model or DEFAULT_MODEL
        key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("backend = \"anthropic\" needs an API key: set ANTHROPIC_API_KEY or llm.api_key")
        self.client = anthropic.Anthropic(api_key=key)

    def describe(self) -> str:
        return f"Claude {self.model} (effort {self.cfg.effort})"

    def warm_up(self) -> None:
        # Fails fast on a bad key or model name instead of on the first question
        try:
            self.client.models.retrieve(self.model)
        except anthropic.AuthenticationError as e:
            raise SystemExit("Anthropic rejected the API key (check ANTHROPIC_API_KEY or llm.api_key)") from e
        except anthropic.NotFoundError as e:
            raise SystemExit(f"Claude model {self.model!r} not found. Set llm.model in config.toml.") from e

    def stream(self, system: str, history: list[dict]) -> Iterator[str]:
        # Leaving the `with` block (including when Chat closes this generator early on an
        # interruption) closes the connection, so Claude stops generating.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            system=system,
            messages=history,
            # Spoken chat doesn't need deep reasoning; low effort keeps time-to-first-word short
            output_config={"effort": self.cfg.effort},
            # If a safety classifier declines, the API retries on Anthropic's recommended fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            spoke = False
            for text in stream.text_stream:
                spoke = True
                yield text
            if stream.get_final_message().stop_reason == "refusal" and not spoke:
                yield "Sorry, I can't help with that one."
