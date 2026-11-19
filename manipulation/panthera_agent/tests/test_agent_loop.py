"""The agent loop with a scripted stand-in for the LLM, and the request the Claude connector sends."""
import json
import re
from types import SimpleNamespace

from core.build import load_config
from core.tasks import TASKS
from llm.claude import ClaudePlanner
from llm.tools import TOOLS
from tools.run_agent import run_one


def _text_of_last_results(messages):
    """All text in the latest user turn (tool results)."""
    out = []
    for block in messages[-1]['content']:
        content = block.get('content') if isinstance(block, dict) else None
        if isinstance(content, list):
            out += [c['text'] for c in content if c.get('type') == 'text']
        elif isinstance(block, dict) and block.get('type') == 'text':
            out.append(block['text'])
    return '\n'.join(out)


class ScriptedPlanner:
    """Plays the model's part: find red, find green, pick red, place on green, check, done."""
    model, effort = 'scripted', None

    def __init__(self):
        self.turn = 0
        self.numbers = {}

    def create(self, system, tools, messages):
        assert system and [t['name'] for t in tools] == [t['name'] for t in TOOLS]
        last = _text_of_last_results(messages)
        for colour in ('red', 'green'):
            m = re.search(rf'#(\d+) cube_{colour}', last)
            if m:
                self.numbers[colour] = int(m.group(1))
        steps = [('find', {'query': 'red cube'}), ('find', {'query': 'green cube'}),
                 ('pick', {'number': self.numbers.get('red', -1)}),
                 ('place_on', {'number': self.numbers.get('green', -1)}),
                 ('find', {'query': 'red cube'}), ('done', {'success': True, 'summary': 'red is on green'})]
        name, args = steps[self.turn]
        self.turn += 1
        call = SimpleNamespace(type='tool_use', id=f'call_{self.turn}', name=name, input=args)
        note = SimpleNamespace(type='text', text=f'turn {self.turn}')
        return SimpleNamespace(content=[note, call], stop_reason='tool_use', usage=None)


def test_scripted_planner_stacks_and_everything_is_logged(tmp_path):
    cfg = load_config()
    task = TASKS['stack_red_on_green']
    out = run_one(task.instruction, task, 2, cfg, 'sim_truth', ScriptedPlanner(), tmp_path)
    assert out['done'] and out['success'] and out['judged_success'], out
    lines = [json.loads(line) for line in (tmp_path / 'steps.jsonl').read_text().splitlines()]
    tools_called = [r.get('tool') for r in lines if r.get('tool')]
    assert tools_called == ['find', 'find', 'pick', 'place_on', 'find', 'done']
    assert all(r['ok'] for r in lines if r.get('tool') not in (None, 'done'))
    assert list(tmp_path.glob('*.jpg')) and (tmp_path / 'outcome.json').exists()


def test_claude_request_shape():
    sent = {}
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: sent.update(kw) or SimpleNamespace(content=[], stop_reason='end_turn'))))
    planner = ClaudePlanner(model='claude-opus-5-5', effort='medium', client=client)
    planner.create('system', TOOLS, [{'role': 'user', 'content': 'hi'}])
    assert sent['model'] == 'claude-opus-5-5' and sent['output_config'] == {'effort': 'medium'}
    assert sent['tool_choice'] == {'type': 'auto', 'disable_parallel_tool_use': True}
    assert sent['fallbacks'] == 'default' and sent['betas'] == ['server-side-fallback-2026-07-01']
    assert sent['cache_control'] == {'type': 'ephemeral'}
    for t in sent['tools']:
        assert t['strict'] and t['input_schema']['additionalProperties'] is False
