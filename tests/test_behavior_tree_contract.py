"""Keeps the behavior tree consistent with the rest of the repo (static, no ROS):
the skill list vs the contract doc, the planner's output, the .msg comments, the
launch file, and the action/topic names the node uses."""
import os
import re

import yaml

from behavior_tree import bt_logic
from behavior_tree.skills import SKILLS
from task_planner_stub.planner_logic import _TASK_KEYWORDS, plan_for_intent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')


def _read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _contract():
    return yaml.safe_load(_read('src', 'humanoid_interfaces', 'config', 'interface_contract.yaml'))


def _documented_skills():
    """Skill names in the 'Skill vocabulary' table of INTERFACE_CONTRACT.md."""
    text = _read('INTERFACE_CONTRACT.md')
    section = text.split('## Skill vocabulary', 1)[1].split('\n## ', 1)[0]
    names = set()
    for line in section.splitlines():
        if line.startswith('|') and not line.startswith('|---') and 'Skill name' not in line:
            names.update(re.findall(r'`([a-z_]+)`', line.split('|')[1]))
    return names


def test_skill_table_matches_the_contract_doc():
    assert _documented_skills() == set(SKILLS)


def test_every_skill_the_planner_can_produce_is_runnable():
    produced = {task for _keyword, task in _TASK_KEYWORDS}
    assert produced <= set(SKILLS), f'planner can emit {produced - set(SKILLS)}, which the BT cannot run'
    cube = {'object_id': 'cube_1', 'label': 'cube', 'x': 2.0, 'y': 0.0, 'z': 0.0}
    for intent in ('pick up the cube', 'put the cube down', 'hand over the cube', 'do something'):
        assert {s['name'] for s in plan_for_intent(intent, [cube])} <= set(SKILLS), intent


def test_skill_msg_comment_lists_every_skill():
    text = _read('src', 'humanoid_interfaces', 'msg', 'Skill.msg')
    for name in SKILLS:
        assert f'"{name}"' in text, f'Skill.msg comment does not mention {name}'


def test_behavior_tree_is_a_real_node_launched_by_a_true_false_switch():
    node = next(n for n in _contract()['nodes'] if n['id'] == 'behavior_tree')
    assert node['status'] == 'real' and node['package'] == 'behavior_tree'
    launch = _read('src', 'bringup', 'launch', 'full_stack.launch.py')
    stub_layers = launch.split('STUB_LAYERS = {', 1)[1].split('}', 1)[0]
    bool_nodes = launch.split('BOOL_NODES = {', 1)[1].split('}', 1)[0]
    assert "'behavior_tree'" not in stub_layers
    assert "'behavior_tree': ('behavior_tree', 'bt_node', 'true')" in bool_nodes
    assert '<exec_depend>behavior_tree</exec_depend>' in _read('src', 'bringup', 'package.xml')


def test_node_uses_the_contracts_action_and_topic_names():
    contract = _contract()
    source = _read('src', 'behavior_tree', 'behavior_tree', 'bt_node.py')
    clients = [a['name'] for a in contract['actions'] if 'behavior_tree' in a['clients']]
    assert sorted(clients) == ['execute_manipulation', 'navigate_to_pose']
    for name in clients:
        assert f"'{name}'" in source, f'bt_node.py does not create an action client for {name}'
    assert "'/skill_sequence'" in source
    owned = [t['name'] for t in contract['topics'] if t['owner'] == 'behavior_tree']
    assert owned == ['/task_status']
    assert "create_publisher(String, '/task_status'" in source


def test_node_has_no_decision_logic_of_its_own():
    """Decisions live in the ROS-free modules (that is what the unit tests cover)."""
    source = _read('src', 'behavior_tree', 'behavior_tree', 'bt_node.py')
    for banned in ('MAX_ATTEMPTS', 'SKILL_TIMEOUT_S', 'GoalStatus', 'time.sleep'):
        assert banned not in source, f'bt_node.py should not contain {banned}; keep logic in bt_logic/action_call'


def test_readme_states_the_same_limits_as_the_code():
    readme = _read('src', 'behavior_tree', 'README.md')
    assert f'`MAX_ATTEMPTS = {bt_logic.MAX_ATTEMPTS}`' in readme
    assert f'`SKILL_TIMEOUT_S = {int(bt_logic.SKILL_TIMEOUT_S)}`' in readme


def test_every_status_the_node_can_publish_is_one_smoke_task_understands():
    """smoke_task.py matches statuses by prefix: RUNNING / SUCCEEDED / FAILED."""
    smoke = _read('src', 'bringup', 'scripts', 'smoke_task.py')
    for prefix in ('RUNNING', 'SUCCEEDED', 'FAILED'):
        assert prefix in smoke
    assert bt_logic.format_status('IDLE') == 'IDLE'
