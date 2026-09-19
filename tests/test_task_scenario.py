"""Cross-stub scenario, pure logic (no ROS). The default demo ('pick up the
cube' with perception's default cube) must be plannable, navigable within
nav_stub's timeout using the same dead-reckoning state_estimation_stub does, and
must stay inside slam_stub's map. Catches the stubs' default constants drifting
apart (e.g. someone moves the cube 20 m away, or changes the standoff)."""
import ast
import os

from behavior_tree_stub.bt_logic import action_kind
from nav_stub.nav_logic import go_to_goal_command, reached, yaw_from_quaternion
from state_estimation_stub.kinematics import integrate_body_velocity, yaw_to_quaternion
from task_planner_stub.planner_logic import plan_for_intent

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')


def _module_constant(pkg, module, name):
    tree = ast.parse(open(os.path.join(SRC, pkg, pkg, module + '.py')).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f'{pkg}.{module}.{name} not found')


def _default_param(pkg, module, param):
    """Reads declare_parameter('<param>', <default>) from a node source."""
    tree = ast.parse(open(os.path.join(SRC, pkg, pkg, module + '.py')).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'declare_parameter'
                and node.args and ast.literal_eval(node.args[0]) == param):
            return ast.literal_eval(node.args[1])
    raise AssertionError(f'{pkg}: no declare_parameter({param!r})')


def test_default_demo_is_plannable_navigable_and_inside_the_map():
    cube = {'object_id': 'cube_1', 'label': 'cube',
            'x': _default_param('perception_stub', 'perception_node', 'cube_x'),
            'y': _default_param('perception_stub', 'perception_node', 'cube_y'),
            'z': _default_param('perception_stub', 'perception_node', 'cube_z')}
    standoff = _default_param('task_planner_stub', 'planner_node', 'standoff_m')
    plan = plan_for_intent('pick up the cube', [cube], standoff)

    assert [action_kind(s['name']) for s in plan] == ['nav', 'manip']
    goal = plan[0]

    half_map = _module_constant('slam_stub', 'slam_node', 'MAP_SIZE_CELLS') * \
        _module_constant('slam_stub', 'slam_node', 'MAP_RESOLUTION') / 2.0
    assert abs(goal['x']) < half_map and abs(goal['y']) < half_map, 'nav goal outside the stub map'

    # closed loop: nav_stub's controller -> state_estimation_stub's dead reckoning
    x = _default_param('state_estimation_stub', 'state_estimation_node', 'initial_x')
    y = _default_param('state_estimation_stub', 'state_estimation_node', 'initial_y')
    yaw = _default_param('state_estimation_stub', 'state_estimation_node', 'initial_yaw')
    dt = 1.0 / _module_constant('nav_stub', 'nav_node', 'CONTROL_RATE_HZ')
    timeout = _module_constant('nav_stub', 'nav_node', 'GOAL_TIMEOUT_S')
    t = 0.0
    while not reached(x, y, goal['x'], goal['y']) and t < timeout:
        # the pose the controller sees is the estimator's (quaternion round trip included)
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        vx, vy, wz = go_to_goal_command(x, y, yaw_from_quaternion(qx, qy, qz, qw), goal['x'], goal['y'])
        x, y, yaw = integrate_body_velocity(x, y, yaw, vx, vy, wz, dt)
        t += dt
    assert reached(x, y, goal['x'], goal['y']), f'did not reach goal within {timeout}s (t={t:.1f})'
    assert t < timeout / 2, f'demo takes {t:.1f}s, uncomfortably close to the {timeout}s timeout'
