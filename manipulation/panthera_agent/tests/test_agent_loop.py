"""The agent loop with a scripted stand-in for the LLM, and the request the Claude connector sends."""
import json
from types import SimpleNamespace

from core.build import load_config
from core.tasks import TASKS
from llm.claude import ClaudePlanner
from llm.scripted import RedOnGreenScript
from llm.tools import TOOLS
from tools.run_agent import run_one


def test_scripted_planner_stacks_and_everything_is_logged(tmp_path):
    cfg = load_config()
    task = TASKS['stack_red_on_green']
    out = run_one(task.instruction, task, 2, cfg, 'sim_truth', RedOnGreenScript(), tmp_path)
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
    assert sent['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
    for t in sent['tools']:
        assert t['strict'] and t['input_schema']['additionalProperties'] is False
