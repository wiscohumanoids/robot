"""The agent loop: the instruction and an overhead image in, then one tool call per turn until `done`.

The conversation is append-only (every reply goes back unchanged, thinking blocks included, and tool results are
only ever appended), which keeps the prompt cache valid turn to turn and is what the API requires for the model's
thinking to carry across turns.

Everything is logged to `log_dir`: steps.jsonl (one line per turn: the model's summarized thinking, the tool call,
its result, token usage) and each image the model was shown. These logs are the debugging record and, later,
training data for a smaller planner.

`on_event(kind, data)`, when given, hears every step as it happens: run_start (system prompt, tools), llm_request
(the content new this turn), llm_response, tool_start, tool_result, run_end. tools/viewer.py shows them live; a
listener may block to pause the run, or raise StopRun to end it.
"""
from __future__ import annotations

import json
import pathlib
import time

import cv2

from core.skills import Result, Skills
from llm.prompts import system_prompt
from llm.tools import TOOLS, content_blocks, run_tool

NUDGE = 'Call one of the tools. When the task is finished, or cannot be finished, call done.'


class StopRun(BaseException):
    """Raised by a listener to end a run. A BaseException so the tool-error handling does not catch it."""


def block_dict(block) -> dict:
    """A response content block (SDK model or plain object) as a dict."""
    if hasattr(block, 'model_dump'):
        return block.model_dump(exclude_none=True)
    return {k: v for k, v in vars(block).items() if v is not None}


def _usage(resp) -> dict:
    u = getattr(resp, 'usage', None)
    if u is None:
        return {}
    return {k: getattr(u, k, None) for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens',
                                             'cache_creation_input_tokens')}


class AgentRun:
    """One instruction carried out through the planner."""

    def __init__(self, skills: Skills, planner, config: dict, log_dir: pathlib.Path, max_turns: int = 40,
                 on_event=None):
        self.skills, self.planner, self.config = skills, planner, config
        self.on_event = on_event
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self.messages: list = []
        self._images = 0

    def _save_images(self, result: Result) -> list[str]:
        names = []
        for caption, img in result.images:
            self._images += 1
            name = f'{self._images:03d}_{caption.split(",")[0].replace(" ", "_")}.jpg'
            cv2.imwrite(str(self.log_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            names.append(name)
        return names

    def _emit(self, kind: str, **data) -> None:
        if self.on_event is not None:
            self.on_event(kind, data)

    def _log(self, record: dict) -> None:
        with open(self.log_dir / 'steps.jsonl', 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def run(self, instruction: str) -> dict:
        """Carry out `instruction`; returns {done, success (the model's claim), summary, turns, stop}."""
        system = system_prompt(self.config)
        self._emit('run_start', instruction=instruction, system=system, tools=TOOLS,
                   model=getattr(self.planner, 'model', None), effort=getattr(self.planner, 'effort', None),
                   request=getattr(self.planner, 'request_options', lambda: {})())
        first = self.skills.look('overhead')
        self.messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'Task: {instruction}'}]
                              + content_blocks(first)})
        self._log(dict(turn=0, instruction=instruction, images=self._save_images(first), state=first.message,
                       model=getattr(self.planner, 'model', None), effort=getattr(self.planner, 'effort', None)))
        outcome = dict(done=False, success=False, summary='', turns=0, stop='max_turns')
        try:
            self._turns(system, outcome)
        except StopRun:
            outcome['stop'] = 'stopped'
        self._log(dict(outcome=outcome, trace=self.skills.trace))
        self._emit('run_end', outcome=outcome)
        return outcome

    def _turns(self, system: str, outcome: dict) -> None:
        for turn in range(1, self.max_turns + 1):
            self._emit('llm_request', turn=turn, messages=len(self.messages), new=self.messages[-1]['content'])
            t0 = time.time()
            resp = self.planner.create(system, TOOLS, self.messages)
            outcome['turns'] = turn
            text = ' '.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
            thinking = ' '.join(getattr(b, 'thinking', '') or '' for b in resp.content
                                if getattr(b, 'type', '') == 'thinking')
            calls = [b for b in resp.content if getattr(b, 'type', '') == 'tool_use']
            record = dict(turn=turn, stop_reason=resp.stop_reason, thinking=thinking, text=text, usage=_usage(resp),
                          model_seconds=round(time.time() - t0, 2))
            self._emit('llm_response', turn=turn, stop_reason=resp.stop_reason, usage=_usage(resp),
                       seconds=record['model_seconds'], content=[block_dict(b) for b in resp.content],
                       model=getattr(resp, 'model', None))
            if resp.stop_reason == 'refusal':
                self._log({**record, 'stop_details': str(getattr(resp, 'stop_details', ''))})
                outcome['stop'] = 'refusal'
                return
            self.messages.append({'role': 'assistant', 'content': resp.content})
            if not calls:
                if resp.stop_reason == 'max_tokens':
                    self._log(record)
                    outcome['stop'] = 'max_tokens'
                    return
                self.messages.append({'role': 'user', 'content': [{'type': 'text', 'text': NUDGE}]})
                self._log(record)
                continue
            results, finished = [], False
            for call in calls:
                args = dict(call.input)
                if call.name == 'done':
                    outcome.update(done=True, success=bool(args.get('success')), summary=str(args.get('summary', '')),
                                   stop='done')
                    results.append({'type': 'tool_result', 'tool_use_id': call.id, 'content': 'Finished.'})
                    self._log({**record, 'tool': 'done', 'input': args})
                    finished = True
                    continue
                self._emit('tool_start', turn=turn, name=call.name, input=args)
                t1 = time.time()
                try:
                    result = run_tool(self.skills, call.name, args)
                except Exception as e:                       # a tool crash is reported to the model, not raised
                    result = Result(False, f'{call.name} raised {type(e).__name__}: {e}')
                results.append({'type': 'tool_result', 'tool_use_id': call.id, 'content': content_blocks(result),
                                **({} if result.ok else {'is_error': True})})
                self._emit('tool_result', turn=turn, name=call.name, ok=result.ok, message=result.message,
                           images=result.images, seconds=round(time.time() - t1, 2))
                self._log({**record, 'tool': call.name, 'input': args, 'ok': result.ok, 'result': result.message,
                           'images': self._save_images(result), 'tool_seconds': round(time.time() - t1, 2),
                           'details': result.details})
            self.messages.append({'role': 'user', 'content': results})
            if finished:
                return
