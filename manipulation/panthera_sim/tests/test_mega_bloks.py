"""Geometry and reproducibility contracts for synthetic block segmentation."""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np

SPEC = importlib.util.spec_from_file_location('mega_bloks', Path(__file__).parents[1] / 'sim/mega_bloks.py')
blocks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(blocks)


def test_geometry_and_semantic_parts(tmp_path):
    path, objects = blocks.build_mega_bloks_scene(tmp_path / 'scene.xml', count=12)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.ncam == 3
    assert model.nu == 14
    for obj in objects:
        gid = model.geom(obj['body_geom']).id
        assert np.allclose(model.geom_size[gid] * 2,
                           [*obj['dimensions_m'][:2], blocks.BODY_HEIGHT_M])
        assert len(obj['stud_geoms']) == np.prod(obj['shape'])
        for name in obj['stud_geoms']:
            stud = model.geom(name).id
            assert model.geom_bodyid[stud] == model.body(obj['body_name']).id
            mesh = model.geom_dataid[stud]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            assert np.isclose(np.ptp(vertices[:, 2]), blocks.STUD_HEIGHT_M)


def test_seed_and_optional_spawn(tmp_path):
    a, _ = blocks.build_mega_bloks_scene(tmp_path / 'a.xml', seed=42)
    b, _ = blocks.build_mega_bloks_scene(tmp_path / 'b.xml', seed=42)
    c, _ = blocks.build_mega_bloks_scene(tmp_path / 'c.xml', seed=43)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    empty, objects = blocks.build_mega_bloks_scene(tmp_path / 'empty.xml', count=0)
    assert not objects
    assert mujoco.MjModel.from_xml_path(str(empty)).nq == 16
