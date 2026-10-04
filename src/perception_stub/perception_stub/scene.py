"""The stub's fake world (no rclpy, unit-testable).

The map is 10 m x 10 m centred on the origin (same as slam_stub), split into a
10 x 10 grid of 1 m cells. Each object sits at the centre of one cell, on a table
at TABLE_Z. Cells are (col, row): col counts +x from the left edge, row counts +y
from the bottom edge, so cell (5, 5) is x 0..1, y 0..1.

The objects are fixed (picked once, not random per launch) so runs and tests are
repeatable. cube_1 is not here: perception_node builds it from its cube_x/cube_y/
cube_z parameters (default 1.5, 0.3 -> cell (6, 5)).
"""
GRID_CELLS = 10
CELL_M = 1.0
ORIGIN_M = -5.0
TABLE_Z = 0.8

# (object_id, label, col, row)
EXTRA_OBJECTS = (
    ('ball_1', 'ball', 3, 7),
    ('cup_1', 'cup', 8, 2),
    ('box_1', 'box', 7, 8),
    ('bottle_1', 'bottle', 2, 2),
)


def cell_center(col, row):
    """Map-frame (x, y) of the centre of a grid cell."""
    return (ORIGIN_M + (col + 0.5) * CELL_M, ORIGIN_M + (row + 0.5) * CELL_M)


def cell_of(x, y):
    """(col, row) of the cell containing the map point (x, y)."""
    return (int((x - ORIGIN_M) // CELL_M), int((y - ORIGIN_M) // CELL_M))


def extra_objects():
    """List of dicts {object_id, label, x, y, z, col, row} for EXTRA_OBJECTS."""
    out = []
    for object_id, label, col, row in EXTRA_OBJECTS:
        x, y = cell_center(col, row)
        out.append({'object_id': object_id, 'label': label, 'x': x, 'y': y,
                    'z': TABLE_Z, 'col': col, 'row': row})
    return out
