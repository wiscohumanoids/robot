"""Keeps the behavior tree consistent with the rest of the repo (static, no ROS):
the skill list vs the contract doc, the planner's output, the .msg comments, the
launch file, and the action/topic names the node uses."""
import os
import re

import yaml

from behavior_tree import bt_logic
from behavior_tree.skills import SKILLS
from task_planner import skill_catalog
from task_planner.planner_logic import _TASK_KEYWORDS, plan_for_intent

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


def test_planner_catalog_and_behavior_tree_skills_are_the_same_set():
    assert set(skill_catalog.names()) == set(SKILLS)


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


def test_plan_id_is_in_the_message_and_used_by_planner_and_tree():
    assert 'string plan_id' in _read('src', 'humanoid_interfaces', 'msg', 'SkillSequence.msg')
    assert 'seq.plan_id = plan_id' in _read('src', 'task_planner', 'task_planner', 'planner_node.py')
    assert 'msg.plan_id' in _read('src', 'behavior_tree', 'behavior_tree', 'bt_node.py')


def test_held_object_topic_has_one_publisher_and_the_planner_reads_it():
    topic = next(t for t in _contract()['topics'] if t['name'] == '/manipulation/held_object')
    assert topic['owner'] == 'manipulation' and topic['consumers'] == ['task_planner']
    assert "create_publisher(String, '/manipulation/held_object'" in _read(
        'src', 'manipulation_runner', 'manipulation_runner', 'manipulation_node.py')
    assert "'/manipulation/held_object'" in _read('src', 'task_planner', 'task_planner', 'planner_node.py')


def test_the_stub_hand_rules_agree_with_the_skill_catalog():
    from manipulation_runner.hand_logic import EMPTY, next_hand
    for name in SKILLS:
        spec = skill_catalog.get(name)
        after = next_hand('x', name, 'cup_1', True)
        assert after == {'take': 'cup_1', 'release': EMPTY, '': 'x'}[spec.hand], name


def test_planner_command_topic_and_the_print_script():
    topic = next(t for t in _contract()['topics'] if t['name'] == '/planner_command')
    assert topic['owner'] == 'speech_input' and topic['consumers'] == ['task_planner']
    assert "'/planner_command'" in _read('src', 'task_planner', 'task_planner', 'planner_node.py')
    script = _read('src', 'bringup', 'scripts', 'watch_status.py')
    assert "'/planner_status'" in script and "'/task_status'" in script
    assert 'create_publisher' not in script
    assert 'scripts/watch_status.py' in _read('src', 'bringup', 'CMakeLists.txt')


def test_the_planner_is_a_real_node_launched_by_a_true_false_switch():
    node = next(n for n in _contract()['nodes'] if n['id'] == 'task_planner')
    assert node['status'] == 'real' and node['package'] == 'task_planner' and node['replaces'] is None
    launch = _read('src', 'bringup', 'launch', 'full_stack.launch.py')
    assert "'task_planner'" not in launch.split('STUB_LAYERS = {', 1)[1].split('}', 1)[0]
    assert "'planner': ('task_planner', 'planner_node', 'true')" in launch
    assert '<exec_depend>task_planner</exec_depend>' in _read('src', 'bringup', 'package.xml')


def test_the_old_stub_package_name_is_gone():
    assert not os.path.exists(os.path.join(SRC, 'task_planner_stub'))
    assert os.path.isdir(os.path.join(SRC, 'task_planner', 'task_planner'))
