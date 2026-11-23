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


class OracleScript:
    """A scripted stand-in for the model on the agent's sim tasks, learning objects included: it answers
    propose_points with the numbered points that lie on the object, read from the simulator's segmentation (which
    the model would judge by looking). For free runs of the whole pipeline in the viewer, and for tests. Sim only.

    `steps`: (tool, args) pairs; an args value may be a callable of the skills, evaluated when the step runs (to
    use a number from the latest find, or points from the latest propose_points)."""

    def __init__(self, skills, steps: list, model: str = 'scripted (sim oracle, no API)'):
        self.skills, self.steps, self.turn = skills, steps, 0
        self.model, self.effort = model, None

    def create(self, system, tools, messages):
        name, args = self.steps[min(self.turn, len(self.steps) - 1)]
        self.turn += 1
        args = {k: (v(self.skills) if callable(v) else v) for k, v in args.items()}
        call = SimpleNamespace(type='tool_use', id=f'call_{self.turn}', name=name, input=args)
        note = SimpleNamespace(type='text', text=f'Scripted step {self.turn}: {name}.')
        return SimpleNamespace(content=[note, call], stop_reason='tool_use', usage=None)


def points_on(body: str):
    """The latest propose_points numbers lying on a sim body (oracle)."""
    def f(skills):
        obs, _, P, _ = skills.proposals['overhead']
        mask = obs.extra['segmentation'] == obs.extra['body_names'].index(body)
        return [c.number for c in P.candidates
                if mask[max(0, int(c.uv[1]) - 4):int(c.uv[1]) + 5, max(0, int(c.uv[0]) - 4):int(c.uv[0]) + 5].mean()
                > 0.3]
    return f


def best(query_word: str):
    """The number of the latest find's best match whose label contains `query_word`."""
    def f(skills):
        objs = [o for o in skills.world.objects.values() if query_word in o.label]
        if not objs:
            return -1                                    # nothing found: the tool says so, as it would to the model
        latest = max(o.step for o in objs)
        return max((o for o in objs if o.step == latest), key=lambda o: (o.trusted, o.score)).number
    return f


ORACLES = {
    'learn_and_stack': [
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'teal puck', 'points': points_on('puck_teal'), 'parts': [], 'deformable': False}),
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'red cube', 'points': points_on('cube_red'), 'parts': [], 'deformable': False}),
        ('find', {'query': 'teal puck'}), ('find', {'query': 'red cube'}),
        ('pick', {'number': best('puck'), 'part': None}), ('place_on', {'number': best('cube')}),
        ('home', {}), ('find', {'query': 'teal puck'}),
        ('done', {'success': True, 'summary': 'the puck is on the cube'})],
    'mug_left': [
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'purple mug', 'points': points_on('mug_purple'), 'parts': [], 'deformable': False}),
        ('find', {'query': 'purple mug'}), ('pick', {'number': best('mug'), 'part': None}),
        ('place_at', {'x_cm': 34.0, 'y_cm': 18.0}), ('home', {}), ('find', {'query': 'purple mug'}),
        ('done', {'success': True, 'summary': 'the mug is on the left'})],
    'peg_insert': [
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'red peg', 'points': points_on('peg_red'), 'parts': [], 'deformable': False}),
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'orange block', 'points': points_on('fixture_orange'), 'parts': [], 'deformable': False}),
        ('find', {'query': 'red peg'}), ('pick', {'number': best('peg'), 'part': 'top'}),
        ('find', {'query': 'orange block'}), ('insert', {'number': best('block')}),
        ('done', {'success': True, 'summary': 'the peg is in the hole'})],
    'pour_beads': [
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'teal cup', 'points': points_on('cup_teal'), 'parts': [], 'deformable': False}),
        ('propose_points', {'camera': 'overhead', 'detail': None}),
        ('learn', {'name': 'blue bowl', 'points': points_on('bowl_blue'), 'parts': [], 'deformable': False}),
        ('find', {'query': 'teal cup'}), ('pick', {'number': best('cup'), 'part': None}),
        ('find', {'query': 'blue bowl'}), ('pour_into', {'number': best('bowl')}),
        ('place_at', {'x_cm': 30.0, 'y_cm': 20.0}), ('home', {}),
        ('done', {'success': True, 'summary': 'poured'})],
}
