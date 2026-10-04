import ast
import os

from perception_stub import scene

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STANDOFF_M = 0.6      # planner's default approach distance, along -x


def _cube_default(param):
    path = os.path.join(SRC, 'perception_stub', 'perception_stub', 'perception_node.py')
    for node in ast.walk(ast.parse(open(path).read())):
        if (isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'declare_parameter'
                and ast.literal_eval(node.args[0]) == param):
            return ast.literal_eval(node.args[1])
    raise AssertionError(param)


def test_cell_center_and_cell_of_round_trip():
    assert scene.cell_center(0, 0) == (-4.5, -4.5)
    assert scene.cell_center(9, 9) == (4.5, 4.5)
    for col in range(scene.GRID_CELLS):
        for row in range(scene.GRID_CELLS):
            assert scene.cell_of(*scene.cell_center(col, row)) == (col, row)


def test_there_are_four_extra_objects_with_unique_ids_labels_and_cells():
    objs = scene.extra_objects()
    assert len(objs) == 4
    for key in ('object_id', 'label'):
        assert len({o[key] for o in objs}) == 4
    assert len({(o['col'], o['row']) for o in objs}) == 4


def test_every_object_is_at_the_centre_of_a_cell_inside_the_grid():
    for o in scene.extra_objects():
        assert 0 <= o['col'] < scene.GRID_CELLS and 0 <= o['row'] < scene.GRID_CELLS
        assert (o['x'], o['y']) == scene.cell_center(o['col'], o['row'])
        assert scene.cell_of(o['x'], o['y']) == (o['col'], o['row'])


def test_no_object_shares_a_cell_with_the_cube_or_the_robot_start():
    cube_cell = scene.cell_of(_cube_default('cube_x'), _cube_default('cube_y'))
    assert cube_cell == (6, 5)
    cells = {(o['col'], o['row']) for o in scene.extra_objects()}
    assert cube_cell not in cells
    assert scene.cell_of(0.0, 0.0) not in cells


def test_every_navigation_standoff_stays_inside_the_map():
    limit = scene.ORIGIN_M + scene.GRID_CELLS * scene.CELL_M
    for o in scene.extra_objects():
        assert scene.ORIGIN_M < o['x'] - STANDOFF_M < limit
        assert scene.ORIGIN_M < o['y'] < limit


def test_the_planner_can_plan_for_each_object_by_label():
    from task_planner.planner_logic import plan_for_intent
    objs = scene.extra_objects()
    for o in objs:
        plan = plan_for_intent(f'pick up the {o["label"]}', objs)
        assert plan[1]['object_id'] == o['object_id']
