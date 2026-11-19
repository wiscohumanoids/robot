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
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


def make_segmenter(name: str, config: dict):
    """'sim_truth' (the sim's masks) or 'server' (the GPU VM's perception server)."""
    if name == 'sim_truth':
        from perception.sim_truth import SimTruthSegmenter
        return SimTruthSegmenter(config['objects'])
    if name == 'server':
        from perception.server_client import ServerSegmenter
        return ServerSegmenter()
    raise ValueError(f'unknown segmenter {name!r} (sim_truth, server)')


def make_skills(robot, segmenter, config: dict):
    """Skills over a fresh World."""
    from core.skills import Skills
    from core.world import World
    return Skills(robot, segmenter, World.from_config(config['objects']), config)
