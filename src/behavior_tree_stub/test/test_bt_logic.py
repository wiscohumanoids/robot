from behavior_tree_stub.bt_logic import action_kind, format_status


def test_navigate_uses_nav_action_everything_else_manipulation():
    assert action_kind('navigate_to') == 'nav'
    assert action_kind('pick') == 'manip'
    assert action_kind('handover') == 'manip'


def test_status_formatting():
    assert format_status('IDLE') == 'IDLE'
    assert format_status('RUNNING', 2, 3, 'pick') == 'RUNNING 2/3 pick'
    assert format_status('FAILED', 1, 2, 'navigate_to', 'goal rejected') == (
        'FAILED 1/2 navigate_to: goal rejected')
