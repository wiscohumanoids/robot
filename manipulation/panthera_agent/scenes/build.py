"""Write the agent's scene files next to panthera_sim's (`../panthera_sim/sim/panthera/agent_<name>.xml`).

    .venv/bin/python -m scenes.build            # all scenes; then each loads in MuJoCo as a check
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scenes.catalog import SCENES, scene_xml  # noqa: E402

SCENE_DIR = ROOT.parent / 'panthera_sim' / 'sim' / 'panthera'


def scene_file(name: str) -> str:
    """The file name panthera_sim loads for an agent scene ('agent_objects.xml')."""
    return f'agent_{name}.xml'


def build(check: bool = True) -> list[pathlib.Path]:
    """Write every scene; with `check`, load each in MuJoCo."""
    out = []
    for name, scene in SCENES.items():
        path = SCENE_DIR / scene_file(name)
        path.write_text(scene_xml(scene))
        out.append(path)
        if check:
            import mujoco
            m = mujoco.MjModel.from_xml_path(str(path))
            print(f'{path.name}: {m.nbody} bodies, {m.ngeom} geoms, {m.nflex} flex')
    return out


if __name__ == '__main__':
    build()
