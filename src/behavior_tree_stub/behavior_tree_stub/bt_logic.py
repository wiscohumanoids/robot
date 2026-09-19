"""Pure skill-dispatch logic for behavior_tree_stub (no rclpy, unit-testable)."""

NAV_SKILLS = {'navigate_to'}


def action_kind(skill_name):
    """Which action serves this skill: 'nav' (NavigateToPose) or 'manip'
    (ExecuteManipulation). Anything that is not a navigation skill is treated
    as a manipulation task name, so adding "handover" etc. needs no change here."""
    return 'nav' if skill_name in NAV_SKILLS else 'manip'


def format_status(state, index=None, total=None, skill=None, detail=''):
    """One-line human/machine-readable status for /task_status, e.g.
    'RUNNING 2/3 pick', 'SUCCEEDED', 'FAILED 2/3 pick: manipulation reported failure'."""
    text = state
    if index is not None and total is not None:
        text += f' {index}/{total}'
    if skill:
        text += f' {skill}'
    if detail:
        text += f': {detail}'
    return text
