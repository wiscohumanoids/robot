"""A scripted stand-in for the model: find red, find green, pick red, place it on green, check, done.

It plays the planner's part with no API call, whatever the instruction says, so the loop, the tools and the viewer
can be exercised for free (tests/test_agent_loop.py, tools/viewer.py's "scripted" planner). It reads the object
numbers from the tool results the way the model would.
"""
from __future__ import annotations

import re
from types import SimpleNamespace


def last_results_text(messages: list) -> str:
    """All text in the latest user turn (the tool results)."""
    out = []
    for block in messages[-1]['content']:
        content = block.get('content') if isinstance(block, dict) else None
        if isinstance(content, list):
            out += [c['text'] for c in content if c.get('type') == 'text']
        elif isinstance(block, dict) and block.get('type') == 'text':
            out.append(block['text'])
    return '\n'.join(out)


class RedOnGreenScript:
    """Stacks red on green in six turns."""
    model, effort = 'scripted: red on green (no API)', None

    def __init__(self):
        self.turn = 0
        self.numbers: dict[str, int] = {}

    def create(self, system, tools, messages):
        """One 'model' turn: the next scripted tool call."""
        last = last_results_text(messages)
        for colour in ('red', 'green'):
            m = re.search(rf'#(\d+) cube_{colour}', last)
            if m:
                self.numbers[colour] = int(m.group(1))
        steps = [('find', {'query': 'red cube'}), ('find', {'query': 'green cube'}),
                 ('pick', {'number': self.numbers.get('red', -1)}),
                 ('place_on', {'number': self.numbers.get('green', -1)}),
                 ('find', {'query': 'red cube'}), ('done', {'success': True, 'summary': 'red is on green'})]
        name, args = steps[min(self.turn, len(steps) - 1)]
        self.turn += 1
        call = SimpleNamespace(type='tool_use', id=f'call_{self.turn}', name=name, input=args)
        note = SimpleNamespace(type='text', text=f'Scripted step {self.turn}: {name}.')
        return SimpleNamespace(content=[note, call], stop_reason='tool_use', usage=None)
