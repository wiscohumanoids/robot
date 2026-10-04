from behavior_tree.skills import SKILLS, Outcome, Skill, run_skill
from bt_fakes import FakeCtx


def test_navigate_to_calls_the_nav_action_with_the_pose():
    ctx = FakeCtx()
    out = run_skill(ctx, Skill('navigate_to', pose='P'), 60.0)
    assert out.ok
    assert ctx.calls == [('navigate', 'P', 60.0)]


def test_pick_place_handover_use_manipulation_with_the_skill_name_as_task():
    for name in ('pick', 'place', 'handover'):
        ctx = FakeCtx()
        run_skill(ctx, Skill(name, 'cube_1', 'P'), 60.0)
        assert ctx.calls == [('manipulate', name, 'cube_1', 'P', 60.0)]


def test_unknown_skill_fails_without_calling_the_robot_and_is_not_retryable():
    ctx = FakeCtx()
    out = run_skill(ctx, Skill('dance'), 60.0)
    assert not out.ok and not out.retryable
    assert out.reason == 'unknown skill: dance'
    assert ctx.calls == []


def test_a_crashing_skill_becomes_a_failed_outcome():
    class Boom(FakeCtx):
        def navigate(self, pose, timeout_s):
            raise RuntimeError('boom')
    out = run_skill(Boom(), Skill('navigate_to'), 60.0)
    assert not out.ok and out.retryable and 'boom' in out.reason


def test_skill_table_covers_the_contract_vocabulary():
    assert set(SKILLS) == {'navigate_to', 'pick', 'place', 'handover'}
