"""First Builders style blocks on the wood bench with both Panthera arms.

Dimensions are configurable visual approximations, not manufacturer CAD or an
interlocking contact model. The body and every stud have separate geom names so
rendered segmentation can distinguish the two surfaces.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

BASE_SCENE = Path(__file__).parent / 'panthera' / 'scene_wood_bimanual.xml'
TABLE_Z = .05
SPAWN_XY_LOW = np.array([.17, -.23])
SPAWN_XY_HIGH = np.array([.48, .23])
# Large toddler blocks: 32 mm grid, 40 mm body, 18.5 mm projecting stud.
PITCH_M = .032
BODY_HEIGHT_M = .040
STUD_RADIUS_M = .011
STUD_HEIGHT_M = .0185
SHAPES = ((1, 1), (2, 1), (3, 1), (2, 2))
COLORS = {'red': (.88, .035, .025, 1), 'blue': (.025, .21, .85, 1),
          'yellow': (1., .77, .015, 1), 'green': (.08, .64, .17, 1),
          'orange': (1., .27, .025, 1), 'purple': (.47, .09, .69, 1)}


def _numbers(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def build_mega_bloks_scene(output_path, *, seed=0, count=8, randomize=True):
    """Write a standalone MJCF scene and return its path and object metadata.

    ``count=0`` keeps the bench empty. A seed chooses colors, yaw and separated
    tabletop positions; ``randomize=False`` uses a fixed grid. Body origins sit
    at the center of the rectangular body. Free joints allow later pose changes.
    Mesh and texture paths are absolute so the XML can live in an output folder.
    """
    if not isinstance(count, int) or not 0 <= count <= 12:
        raise ValueError('count must be an integer from 0 through 12')
    rng = np.random.default_rng(seed)
    color_names = tuple(COLORS)
    placements, placed = [], []
    for i in range(count):
        nx, ny = SHAPES[i % len(SHAPES)]
        length, width = nx * PITCH_M - .0004, ny * PITCH_M - .0004
        radius = np.hypot(length, width) / 2
        if randomize:
            for _ in range(2000):
                xy = rng.uniform(SPAWN_XY_LOW, SPAWN_XY_HIGH)
                if all(np.linalg.norm(xy - other) > radius + r + .006 for other, r in placed):
                    break
            else:
                raise ValueError(f'Cannot place {count} separated blocks; reduce count')
            yaw = float(rng.uniform(-np.pi, np.pi))
            color = color_names[int(rng.integers(len(color_names)))]
        else:
            xy = np.array([.19 + .105 * (i % 3), -.22 + .105 * (i // 3)])
            yaw, color = 0., color_names[i % len(color_names)]
        placed.append((xy, radius))
        placements.append((nx, ny, [*xy, TABLE_Z + BODY_HEIGHT_M / 2 + .0002], yaw, color))
    root, metadata = _build_xml(placements)
    return _write_scene(root, output_path), metadata


def _build_xml(placements):
    root = ET.parse(BASE_SCENE).getroot()
    root.set('model', 'panthera_wood_mega_bloks')
    compiler = root.find('compiler')
    compiler.set('meshdir', str(BASE_SCENE.parent.resolve()))
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            asset.set('file', str((BASE_SCENE.parent / asset.get('file')).resolve()))
    # Four radial rings give the projecting stud a slight mould draft and a
    # bevel around its top. MuJoCo builds the convex hull of these vertices.
    vertices = []
    for z, radius in ((0., STUD_RADIUS_M), (.001, STUD_RADIUS_M),
                      (STUD_HEIGHT_M - .0012, STUD_RADIUS_M * .94),
                      (STUD_HEIGHT_M, STUD_RADIUS_M * .84)):
        for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            vertices.extend((radius * np.cos(angle), radius * np.sin(angle), z))
    ET.SubElement(root.find('asset'), 'mesh', name='mega_stud_tapered',
                  vertex=_numbers(vertices))
    world = root.find('worldbody')
    metadata = []
    home = root.find("keyframe/key[@name='home']")
    home_poses = [home.get('qpos')]
    for i, (nx, ny, position, yaw, color) in enumerate(placements):
        length, width = nx * PITCH_M - .0004, ny * PITCH_M - .0004
        name = f'mega_block_{i:03d}'
        body = ET.SubElement(world, 'body', name=name,
                             pos=_numbers(position),
                             quat=_numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        ET.SubElement(body, 'freejoint', name=name)
        # Blocks follow both arms in worldbody, so their seven free-joint
        # coordinates follow the arm positions in the bench's home keyframe.
        home_poses.extend((body.get('pos'), body.get('quat')))
        material = f'mega_plastic_{color}'
        if root.find(f"asset/material[@name='{material}']") is None:
            ET.SubElement(root.find('asset'), 'material', name=material,
                          rgba=_numbers(COLORS[color]), specular='.45', shininess='.5')
        common = dict(material=material, condim='4', friction='.7 .02 .001',
                      contype='1', conaffinity='3', group='2')
        ET.SubElement(body, 'geom', name=name + '_body', type='box',
                      size=_numbers([length / 2, width / 2, BODY_HEIGHT_M / 2]),
                      mass=str(.007 * nx * ny), **common)
        studs = []
        for y in range(ny):
            for x in range(nx):
                stud = f'{name}_stud_{len(studs)}'
                studs.append(stud)
                ET.SubElement(body, 'geom', name=stud, type='mesh', mesh='mega_stud_tapered',
                              pos=_numbers([(x - (nx - 1) / 2) * PITCH_M,
                                            (y - (ny - 1) / 2) * PITCH_M,
                                            BODY_HEIGHT_M / 2]),
                              mass='.001', **common)
        metadata.append(dict(body_name=name, body_geom=name + '_body', stud_geoms=studs,
                             color=color, shape=[nx, ny], dimensions_m=[length, width,
                             BODY_HEIGHT_M + STUD_HEIGHT_M], body_height_m=BODY_HEIGHT_M,
                             stud_height_m=STUD_HEIGHT_M, stud_radius_m=STUD_RADIUS_M))
    home.set('qpos', ' '.join(home_poses))
    return root, metadata


def _write_scene(root, output_path):
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


if __name__ == '__main__':
    import argparse
    import mujoco
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/panthera_mega_bloks.xml'))
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--view', action='store_true', help='Open the scene in the MuJoCo viewer')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    scene, _ = build_mega_bloks_scene(args.output, count=args.count, seed=args.seed)
    print(scene)
    if args.preview or args.view:
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        mujoco.mj_forward(model, data)
    if args.preview:
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            renderer.update_scene(data, camera='overhead')
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.preview)
        print(args.preview.resolve())
    if args.view:
        import mujoco.viewer
        mujoco.viewer.launch(model, data)
