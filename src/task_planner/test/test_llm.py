"""LLM planner: config/.env loading, the OpenAI call (fake transport), and the
validation of whatever the model says. No network, no ROS."""
import json

import pytest

from task_planner import llm_config, llm_planner, skill_catalog
from task_planner.llm_client import LLMError, chat_json
from task_planner.llm_config import LLMConfig

CUP = {'object_id': 'cup_1', 'label': 'cup', 'x': 3.5, 'y': -2.5, 'z': 0.8}
CUBE = {'object_id': 'cube_1', 'label': 'cube', 'x': 1.5, 'y': 0.3, 'z': 0.8}
OBJECTS = [CUBE, CUP]
CONFIG = LLMConfig(api_key='sk-test')


def reply(feasible=True, reason='', steps=()):
    return {'feasible': feasible, 'reason': reason, 'steps': list(steps)}


def step(skill, object_id='', x=None, y=None):
    return {'skill': skill, 'object_id': object_id, 'x': x, 'y': y}


def fake_post(content, capture=None):
    """A transport that answers with a chat completion holding `content`."""
    def post(url, headers, body, timeout_s):
        if capture is not None:
            capture.update(url=url, headers=headers, body=json.loads(body), timeout_s=timeout_s)
        text = content if isinstance(content, str) else json.dumps(content)
        return json.dumps({'choices': [{'message': {'content': text}}]}).encode()
    return post


# ------------------------------------------------------------------ config / .env
def test_parse_env_file():
    text = '# c\n\nOPENAI_API_KEY="sk-1"\nexport OPENAI_MODEL=gpt-x\nBAD LINE\nEMPTY=\nQ=\'a b\'\n'
    assert llm_config.parse_env_file(text) == {
        'OPENAI_API_KEY': 'sk-1', 'OPENAI_MODEL': 'gpt-x', 'EMPTY': '', 'Q': 'a b'}


def test_defaults_without_a_key():
    cfg = llm_config.config_from_env({})
    assert not cfg.configured
    assert cfg.model == llm_config.DEFAULT_MODEL and cfg.base_url == llm_config.DEFAULT_BASE_URL


def test_config_reads_every_variable_and_trims():
    cfg = llm_config.config_from_env({
        'OPENAI_API_KEY': ' sk-1 ', 'OPENAI_MODEL': 'm', 'OPENAI_BASE_URL': 'http://x/v1/', 'LLM_TIMEOUT_S': '5'})
    assert (cfg.api_key, cfg.model, cfg.base_url, cfg.timeout_s) == ('sk-1', 'm', 'http://x/v1', 5.0)
    assert cfg.configured


def test_bad_timeout_falls_back_to_default():
    assert llm_config.config_from_env({'LLM_TIMEOUT_S': 'soon'}).timeout_s == llm_config.DEFAULT_TIMEOUT_S


def test_env_file_is_found_in_a_parent_directory_and_does_not_override(tmp_path):
    (tmp_path / '.env').write_text('OPENAI_API_KEY=from-file\nOPENAI_MODEL=file-model\n')
    sub = tmp_path / 'a' / 'b'
    sub.mkdir(parents=True)
    env = {'OPENAI_MODEL': 'from-env'}
    assert llm_config.load_env(env, str(sub)) == str(tmp_path / '.env')
    assert env == {'OPENAI_MODEL': 'from-env', 'OPENAI_API_KEY': 'from-file'}


# ------------------------------------------------------------------ the call
def test_chat_json_sends_a_strict_schema_request_and_parses_the_reply():
    seen = {}
    out = chat_json(CONFIG, [{'role': 'user', 'content': 'hi'}], llm_planner.SCHEMA,
                    post=fake_post(reply(), seen))
    assert out == reply()
    assert seen['url'] == 'https://api.openai.com/v1/chat/completions'
    assert seen['headers']['Authorization'] == 'Bearer sk-test'
    body = seen['body']
    assert body['model'] == llm_config.DEFAULT_MODEL == 'gpt-5.6-terra'
    assert 'temperature' not in body        # newer models only accept their default
    assert body['response_format']['json_schema']['strict'] is True
    assert body['messages'] == [{'role': 'user', 'content': 'hi'}]
    assert seen['timeout_s'] == llm_config.DEFAULT_TIMEOUT_S


def test_chat_json_without_a_key_does_not_call_out():
    def boom(*a):
        raise AssertionError('must not be called')
    with pytest.raises(LLMError, match='OPENAI_API_KEY'):
        chat_json(LLMConfig(), [], {}, post=boom)


@pytest.mark.parametrize('raw, text', [
    (b'not json', 'not a chat completion'),
    (b'{"choices": []}', 'not a chat completion'),
    (json.dumps({'choices': [{'message': {'content': 'nope'}}]}).encode(), 'not valid JSON'),
    (json.dumps({'choices': [{'message': {'content': None}}]}).encode(), 'not valid JSON'),
    (json.dumps({'choices': [{'message': {'refusal': 'no'}}]}).encode(), 'refused'),
])
def test_unusable_replies_raise_llm_error(raw, text):
    with pytest.raises(LLMError, match=text):
        chat_json(CONFIG, [], {}, post=lambda *a: raw)


# ------------------------------------------------------------------ the prompt
def test_schema_skill_enum_is_the_catalog():
    enum = llm_planner.SCHEMA['properties']['steps']['items']['properties']['skill']['enum']
    assert enum == skill_catalog.names()


def test_prompt_contains_the_catalog_the_map_and_the_objects():
    system, user = llm_planner.build_messages('can you pick up the cup', OBJECTS)
    assert system['role'] == 'system' and user['role'] == 'user'
    assert skill_catalog.describe() in system['content']
    assert '-5' in system['content'] and '5' in system['content']
    assert 'cup_1 (cup) at x=3.50, y=-2.50' in user['content']
    assert 'Request: can you pick up the cup' in user['content']


def test_prompt_tells_the_model_where_the_robot_is_now():
    import math
    robot = {'x': 1.5, 'y': -2.0, 'yaw': math.pi / 2}
    _, user = llm_planner.build_messages('go', OBJECTS, robot=robot)
    assert 'Robot now: x=1.50, y=-2.00, facing 90 degrees' in user['content']
    _, unknown = llm_planner.build_messages('go', OBJECTS)
    assert 'Robot position: unknown.' in unknown['content']


def test_plan_with_llm_sends_the_robot_pose_in_the_request():
    seen = {}
    llm_planner.plan_with_llm('go', OBJECTS, CONFIG, robot={'x': 4.0, 'y': 1.0, 'yaw': 0.0},
                              post=fake_post(reply(False, 'x'), seen))
    assert 'x=4.00, y=1.00' in seen['body']['messages'][1]['content']


def test_prompt_with_no_objects_says_so():
    assert 'none' in llm_planner.describe_objects([])


def test_map_bounds_match_perception_scene():
    from perception_stub import scene
    assert llm_planner.MAP_MIN_M == scene.ORIGIN_M
    assert llm_planner.MAP_MAX_M == scene.ORIGIN_M + scene.GRID_CELLS * scene.CELL_M


# ------------------------------------------------------------------ valid plans
def test_pick_up_the_cup_fills_the_object_pose_from_the_world():
    result = llm_planner.build_plan(reply(steps=[
        step('navigate_to', x=2.9, y=-2.5), step('pick', 'cup_1')]), OBJECTS)
    assert result.ok
    assert result.plan == [
        {'name': 'navigate_to', 'object_id': '', 'x': 2.9, 'y': -2.5, 'z': 0.0},
        {'name': 'pick', 'object_id': 'cup_1', 'x': 3.5, 'y': -2.5, 'z': 0.8}]


def test_go_to_coordinates():
    result = llm_planner.build_plan(reply(steps=[step('navigate_to', x=2, y=-3)]), OBJECTS)
    assert result.plan == [{'name': 'navigate_to', 'object_id': '', 'x': 2.0, 'y': -3.0, 'z': 0.0}]


def test_the_world_pose_wins_over_numbers_the_llm_put_on_an_object_skill():
    result = llm_planner.build_plan(reply(steps=[step('pick', 'cup_1', x=0.0, y=0.0)]), OBJECTS)
    assert (result.plan[0]['x'], result.plan[0]['y']) == (3.5, -2.5)


def test_every_plan_step_passes_the_catalog_check():
    result = llm_planner.build_plan(reply(steps=[
        step('navigate_to', x=1, y=1), step('pick', 'cube_1'), step('handover', 'cube_1')]), OBJECTS)
    assert all(skill_catalog.check_step(s) == [] for s in result.plan)


# ------------------------------------------------------------------ rejected plans
@pytest.mark.parametrize('bad, text', [
    (None, 'not an object'),
    ('plan', 'not an object'),
    (reply(False, 'no such object'), 'cannot do that: no such object'),
    (reply(False, ''), 'no reason given'),
    (reply(True, steps=[]), 'no steps'),
    ({'feasible': True}, 'no steps'),
    (reply(steps=[step('navigate_to', x=1, y=1)] * 11), 'too long'),
    (reply(steps=[step('dance')]), 'unknown skill'),
    (reply(steps=[step('pick')]), 'needs an object_id'),
    (reply(steps=[step('pick', 'ball_9')]), "unknown object 'ball_9'"),
    (reply(steps=[step('navigate_to')]), 'needs a target pose'),
    (reply(steps=[step('navigate_to', x=1)]), 'needs a target pose'),
    (reply(steps=[step('navigate_to', 'cup_1', x=1, y=1)]), 'takes no object_id'),
    (reply(steps=[step('navigate_to', x=50, y=0)]), 'outside the map'),
    (reply(steps=[step('navigate_to', x=0, y=-4.9)]), 'outside the map'),
])
def test_bad_plans_are_rejected_with_a_reason(bad, text):
    result = llm_planner.build_plan(bad, OBJECTS)
    assert not result.ok and result.plan == [] and text in result.error


def test_one_bad_step_rejects_the_whole_plan():
    result = llm_planner.build_plan(reply(steps=[step('navigate_to', x=1, y=1), step('pick', 'ghost')]), OBJECTS)
    assert result.plan == [] and 'step 2' in result.error


def test_navigating_right_up_to_the_edge_margin_is_allowed():
    edge = llm_planner.MAP_MAX_M - llm_planner.EDGE_MARGIN_M
    assert llm_planner.build_plan(reply(steps=[step('navigate_to', x=edge, y=-edge)]), OBJECTS).ok


# ------------------------------------------------------------------ end to end (fake transport)
def test_plan_with_llm_happy_path():
    result = llm_planner.plan_with_llm(
        'can you pick up the cup', OBJECTS, CONFIG,
        post=fake_post(reply(steps=[step('navigate_to', x=2.9, y=-2.5), step('pick', 'cup_1')])))
    assert result.ok and [s['name'] for s in result.plan] == ['navigate_to', 'pick']


def test_plan_with_llm_turns_transport_errors_into_a_result():
    def down(*a):
        raise LLMError('LLM request failed: boom')
    result = llm_planner.plan_with_llm('x', OBJECTS, CONFIG, post=down)
    assert not result.ok and 'boom' in result.error


def test_plan_with_llm_without_a_key_is_an_error_not_a_crash():
    result = llm_planner.plan_with_llm('x', OBJECTS, LLMConfig())
    assert not result.ok and 'OPENAI_API_KEY' in result.error


def test_plan_with_llm_infeasible_request():
    result = llm_planner.plan_with_llm(
        'go to x=50 y=50', OBJECTS, CONFIG, post=fake_post(reply(False, 'outside the map')))
    assert not result.ok and 'outside the map' in result.error


# ------------------------------------------------------------------ replanning
OLD_PLAN = [
    {'name': 'navigate_to', 'object_id': '', 'x': 2.9, 'y': -2.5, 'z': 0.0},
    {'name': 'pick', 'object_id': 'cup_1', 'x': 3.5, 'y': -2.5, 'z': 0.8},
]
FAILURE = 'FAILED 2/2 pick: action aborted (attempt 3/3)'


def test_parse_failure():
    assert llm_planner.parse_failure(FAILURE) == {
        'index': 2, 'total': 2, 'skill': 'pick', 'reason': 'action aborted (attempt 3/3)'}
    assert llm_planner.parse_failure('FAILED 1/3 navigate_to')['reason'] == ''
    assert llm_planner.parse_failure('FAILED: unexpected error: x') is None
    assert llm_planner.parse_failure('SUCCEEDED') is None


def test_parse_failure_understands_everything_the_tree_publishes():
    from behavior_tree.bt_logic import format_status
    text = format_status('FAILED', 3, 4, 'handover', 'timed out after 60 s (attempt 3/3)')
    assert llm_planner.parse_failure(text)['skill'] == 'handover'
    assert llm_planner.parse_failure(format_status('FAILED', 1, 2, 'dance', 'unknown skill: dance'))['skill'] == 'dance'


def test_describe_plan_numbers_the_steps():
    assert llm_planner.describe_plan(OLD_PLAN) == '1. navigate_to x=2.90, y=-2.50\n2. pick cup_1'


def test_describe_failure_says_which_steps_succeeded():
    text = llm_planner.describe_failure(FAILURE)
    assert 'failed at step 2 of 2 (pick): action aborted' in text and 'Steps 1 to 1 succeeded' in text
    assert 'succeeded' not in llm_planner.describe_failure('FAILED 1/2 navigate_to: x (attempt 3/3)')
    assert llm_planner.describe_failure('FAILED: boom') == 'The plan failed: FAILED: boom'


def test_replan_prompt_has_request_old_plan_failure_and_the_current_world():
    robot = {'x': 3.0, 'y': -2.0, 'yaw': 0.0}
    system, user = llm_planner.build_replan_messages(
        'pick up the cup', OLD_PLAN, FAILURE, OBJECTS, robot=robot)
    assert skill_catalog.describe() in system['content']
    text = user['content']
    assert 'Original request: pick up the cup' in text
    assert '1. navigate_to x=2.90, y=-2.50\n2. pick cup_1' in text
    assert 'failed at step 2 of 2 (pick)' in text
    assert 'Robot now: x=3.00, y=-2.00' in text and 'cup_1 (cup)' in text
    assert llm_planner.REPLAN_RULES in text


def test_replan_with_llm_returns_a_validated_new_plan():
    seen = {}
    new = reply(steps=[step('navigate_to', x=3.0, y=-2.0), step('pick', 'cup_1')])
    result = llm_planner.replan_with_llm('pick up the cup', OLD_PLAN, FAILURE, OBJECTS, CONFIG,
                                         post=fake_post(new, seen))
    assert result.ok and [s['name'] for s in result.plan] == ['navigate_to', 'pick']
    assert 'Previous plan:' in seen['body']['messages'][1]['content']


def test_the_model_may_use_any_catalog_skill_when_replanning():
    new = reply(steps=[step('navigate_to', x=1, y=1), step('pick', 'cube_1'),
                       step('place', 'cube_1'), step('pick', 'cube_1'), step('handover', 'cube_1')])
    assert llm_planner.replan_with_llm('x', OLD_PLAN, FAILURE, OBJECTS, CONFIG, post=fake_post(new)).ok


def test_an_invalid_replan_is_rejected_like_any_plan():
    new = reply(steps=[step('pick', 'ghost_9')])
    result = llm_planner.replan_with_llm('x', OLD_PLAN, FAILURE, OBJECTS, CONFIG, post=fake_post(new))
    assert not result.ok and 'unknown object' in result.error


def test_a_replan_the_model_declines_has_its_reason():
    result = llm_planner.replan_with_llm('x', OLD_PLAN, FAILURE, OBJECTS, CONFIG,
                                         post=fake_post(reply(False, 'the cup fell off the table')))
    assert not result.ok and 'cup fell' in result.error


def test_replan_transport_errors_become_a_result():
    def down(*a):
        raise LLMError('LLM request failed: boom')
    assert 'boom' in llm_planner.replan_with_llm('x', OLD_PLAN, FAILURE, OBJECTS, CONFIG, post=down).error


# ------------------------------------------------------------------ hand state
@pytest.mark.parametrize('held, text', [
    (None, 'Hand: unknown.'), ('', 'Hand: empty.'), ('cup_1', 'Hand: holding cup_1.')])
def test_describe_hand(held, text):
    assert llm_planner.describe_hand(held) == text


def test_both_prompts_tell_the_model_what_the_hand_holds():
    _, user = llm_planner.build_messages('put it down', OBJECTS, held='cup_1')
    assert 'Hand: holding cup_1.' in user['content']
    _, again = llm_planner.build_replan_messages('x', OLD_PLAN, FAILURE, OBJECTS, held='')
    assert 'Hand: empty.' in again['content']
    _, unknown = llm_planner.build_messages('x', OBJECTS)
    assert 'Hand: unknown.' in unknown['content']


def test_the_system_prompt_states_the_hand_rules():
    text = llm_planner.system_prompt()
    assert 'pick needs an empty hand' in text and 'authoritative' in text


def test_plan_that_picks_while_holding_is_rejected_with_the_reason():
    bad = reply(steps=[step('pick', 'cup_1')])
    result = llm_planner.build_plan(bad, OBJECTS, held='cube_1')
    assert not result.ok and 'hand is already holding cube_1' in result.error


def test_plan_that_places_something_not_held_is_rejected():
    result = llm_planner.build_plan(reply(steps=[step('place', 'cup_1')]), OBJECTS, held='')
    assert not result.ok and 'not holding cup_1 (hand is empty)' in result.error


def test_put_it_down_while_holding_the_cup_is_fine():
    result = llm_planner.build_plan(
        reply(steps=[step('navigate_to', x=1, y=1), step('place', 'cup_1')]), OBJECTS, held='cup_1')
    assert result.ok


def test_an_unknown_hand_does_not_block_a_plan():
    assert llm_planner.build_plan(reply(steps=[step('place', 'cup_1')]), OBJECTS).ok


def test_plan_with_llm_and_replan_pass_the_hand_state_through():
    seen = {}
    llm_planner.plan_with_llm('x', OBJECTS, CONFIG, held='cup_1', post=fake_post(reply(False, 'n'), seen))
    assert 'Hand: holding cup_1.' in seen['body']['messages'][1]['content']
    bad = reply(steps=[step('pick', 'cup_1')])
    result = llm_planner.replan_with_llm('x', OLD_PLAN, FAILURE, OBJECTS, CONFIG, held='ball_1',
                                         post=fake_post(bad))
    assert not result.ok and 'already holding ball_1' in result.error


def test_open_ai_key_is_accepted_as_an_alias_and_the_real_name_wins():
    assert llm_config.config_from_env({'OPEN_AI_KEY': 'sk-alias'}).api_key == 'sk-alias'
    both = {'OPENAI_API_KEY': 'sk-real', 'OPEN_AI_KEY': 'sk-alias'}
    assert llm_config.config_from_env(both).api_key == 'sk-real'


def test_spaces_around_equals_in_the_env_file_are_fine():
    assert llm_config.parse_env_file('OPEN_AI_KEY = sk-1\n') == {'OPEN_AI_KEY': 'sk-1'}
