"""Putting the pieces together for the tools: config, robot, segmenter, skills."""
from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def load_config(path: str | pathlib.Path = ROOT / 'config' / 'sim.yaml') -> dict:
    """A config file (relative paths are from panthera_agent/)."""
    p = pathlib.Path(path)
    return yaml.safe_load(open(p if p.is_absolute() else ROOT / p))


def load_dotenv(path: pathlib.Path = REPO / '.env') -> None:
    """Fill os.environ from the repo's .env (KEY=value lines) without overriding what is already set."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            v = v.strip()
            if v[:1] in ('"', "'") and v[0] in v[1:]:          # quoted: up to the closing quote
                v = v[1:v.index(v[0], 1)]
            else:                                               # unquoted: drop an inline " # comment"
                v = v.split(' #', 1)[0].split('\t#', 1)[0].strip()
            if v:
                os.environ.setdefault(k.strip(), v)


# Where object masks come from: the sim's own (perfect) ones, or one of the GPU VM's perception servers.
SEGMENTERS = {'sim_truth': 'MuJoCo ground truth', 'dino_local': 'DINOv3, learned objects (this machine)',
              'dino_sam2': 'Grounding DINO + SAM 2.1 (VM)', 'sam3': 'SAM 3 (VM)'}


def make_segmenter(name: str, config: dict, memory=None):
    """One of SEGMENTERS. `memory`: the seen-objects list (perception/memory.py) dino_local finds objects from."""
    if name == 'sim_truth':
        from perception.sim_truth import SimTruthSegmenter
        return SimTruthSegmenter(config['objects'])
    if name == 'dino_local':
        from perception.dino_local import DinoLocalSegmenter
        return DinoLocalSegmenter(memory, **(config.get('segmenters') or {}).get('dino_local', {}))
    if name in ('dino_sam2', 'sam3'):
        from perception.server_client import ServerSegmenter
        return ServerSegmenter(name, **(config.get('segmenters') or {}).get(name, {}))
    raise ValueError(f'unknown segmenter {name!r} ({", ".join(SEGMENTERS)})')


def make_skills(robot, segmenter, config: dict, memory=None):
    """Skills over a fresh World, with the seen-objects list (the segmenter's, `memory`, or the default folder)."""
    from core.skills import Skills
    from core.world import World
    return Skills(robot, segmenter, World.from_config(config['objects']), config, memory=memory)


def config_for_scene(config: dict, scene: str) -> dict:
    """The config with the scene's file and, for the agent's scenes (scenes/catalog.py), its table extent."""
    out = {**config, 'scene': scene}
    if scene.startswith('agent_'):
        from scenes.catalog import SCENES
        sc = SCENES[scene[len('agent_'):-len('.xml')]]
        out['table'] = {**config['table'], 'x': list(sc.table_x), 'y': list(sc.table_y)}
    return out
