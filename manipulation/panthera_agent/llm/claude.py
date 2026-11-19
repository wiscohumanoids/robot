"""The planner's connection to Claude (Anthropic Messages API, official Python SDK).

Model and effort come from the environment (or arguments): PANTHERA_AGENT_MODEL (default claude-opus-5-5) and
PANTHERA_AGENT_EFFORT (default medium). The key is ANTHROPIC_API_KEY, from the repo's .env or the environment.

Every request:
- one tool call per turn (`disable_parallel_tool_use`): the robot does one thing, then the model sees what happened;
- adaptive thinking (the model's default), effort from `output_config`;
- prompt caching on the request (the system prompt and tool list stay identical, so each turn re-reads the
  growing history from the cache);
- server-side refusal fallback (`fallbacks: "default"`): if a safety classifier declines a turn, the API re-runs it
  on a fallback model in the same call instead of stopping the robot mid-task.
"""
from __future__ import annotations

import os

DEFAULT_MODEL = 'claude-opus-5-5'
DEFAULT_EFFORT = 'medium'
FALLBACK_BETA = 'server-side-fallback-2026-07-01'


class ClaudePlanner:
    """Sends the conversation and tools to Claude, returns its reply."""

    def __init__(self, model: str | None = None, effort: str | None = None, client=None, max_tokens: int = 16000,
                 fallback: bool = True):
        self.model = model or os.environ.get('PANTHERA_AGENT_MODEL') or DEFAULT_MODEL
        self.effort = effort or os.environ.get('PANTHERA_AGENT_EFFORT') or DEFAULT_EFFORT
        self.max_tokens = max_tokens
        self.fallback = fallback
        if client is None:
            import anthropic
            if not os.environ.get('ANTHROPIC_API_KEY'):
                raise RuntimeError('ANTHROPIC_API_KEY is not set: add it to the repo\'s .env (see .env.example)')
            client = anthropic.Anthropic()
        self.client = client

    def create(self, system: str, tools: list[dict], messages: list):
        """One model turn."""
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=messages,
                  tool_choice={'type': 'auto', 'disable_parallel_tool_use': True},
                  output_config={'effort': self.effort}, cache_control={'type': 'ephemeral'})
        if self.fallback:
            kw.update(betas=[FALLBACK_BETA], fallbacks='default')
        return self.client.beta.messages.create(**kw)
