"""Episodes of the local skills, on disk: what every training step reads and every practice run writes.

One episode = one attempt at a local skill, from the handoff to the end:

    data/episodes/<skill>/<id>.npz   wrist (T, 64, 64, 3) uint8, state (T, 5), action (T, 5) as executed,
                                     expert (T, 5) the supervisor's action where it gave one (NaN elsewhere),
                                     intervened (T,) bool (the supervisor drove that step), reward (T,)
    data/episodes/<skill>/index.jsonl   one line per episode: id, source (expert | scripted | policy | dagger |
                                     rl | real), success, how success was judged (truth | classifier | claude |
                                     human | check), steps, interventions, the policy version, labels added later
                                     (Claude's verdict), and free notes

The same format on the sim and on the rig, so the nightly job (learning/nightly.sh) trains on both. `filter` picks
the episodes a training run uses: successful ones for imitation, all of them for RL.

Data stays out of the repository (data/ is ignored); tools/vm.sh-style rsync moves it to the GPU VM.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def data_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get('PANTHERA_AGENT_DATA') or ROOT / 'data')


@dataclass
class Episode:
    skill: str
    wrist: list = field(default_factory=list)
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    expert: list = field(default_factory=list)
    intervened: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, obs: dict, action, expert=None, intervened: bool = False, reward: float = 0.0) -> None:
        self.wrist.append(np.asarray(obs['wrist'], np.uint8))
        self.state.append(np.asarray(obs['state'], np.float32))
        self.action.append(np.asarray(action, np.float32))
        self.expert.append(np.full(5, np.nan, np.float32) if expert is None else np.asarray(expert, np.float32))
        self.intervened.append(bool(intervened))
        self.reward.append(float(reward))

    def __len__(self) -> int:
        return len(self.action)

    def arrays(self) -> dict:
        return dict(wrist=np.stack(self.wrist), state=np.stack(self.state), action=np.stack(self.action),
                    expert=np.stack(self.expert), intervened=np.array(self.intervened, bool),
                    reward=np.array(self.reward, np.float32))


class EpisodeStore:
    """Episodes of one skill on disk."""

    def __init__(self, skill: str, root: pathlib.Path | None = None):
        self.skill = skill
        self.folder = (root or data_root()) / 'episodes' / skill
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / 'index.jsonl'

    def save(self, ep: Episode, views: dict | None = None, **meta) -> str:
        """Store an episode; `views`: full-resolution images of how it ended ({'wrist': rgb, 'overhead': rgb}),
        kept as JPEGs next to it for labelling (a 64-pixel frame cannot show whether a peg went in)."""
        if not len(ep):
            return ''
        eid = time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns() % 1_000_000:06d}'
        np.savez_compressed(self.folder / f'{eid}.npz', **ep.arrays())
        if views:
            import cv2
            for name, img in views.items():
                cv2.imwrite(str(self.folder / f'{eid}_{name}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
        line = dict(id=eid, skill=self.skill, steps=len(ep), interventions=int(sum(ep.intervened)),
                    time=time.time(), **ep.meta, **meta)
        with open(self.index, 'a') as f:
            f.write(json.dumps(line, default=str) + '\n')
        return eid

    def entries(self) -> list[dict]:
        if not self.index.exists():
            return []
        out = {}
        for line in self.index.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                out[j['id']] = {**out.get(j['id'], {}), **j}         # later lines (labels) update earlier ones
        return list(out.values())

    def label(self, eid: str, **fields) -> None:
        """Add fields to an episode's index entry (an appended line; the latest wins)."""
        with open(self.index, 'a') as f:
            f.write(json.dumps(dict(id=eid, **fields), default=str) + '\n')

    def load(self, eid: str) -> dict:
        z = np.load(self.folder / f'{eid}.npz')
        return {k: z[k] for k in z.files}

    def views(self, eid: str) -> dict:
        import cv2
        out = {}
        for f in self.folder.glob(f'{eid}_*.jpg'):
            out[f.stem[len(eid) + 1:]] = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        return out

    def filter(self, success: bool | None = None, sources: set[str] | None = None,
               trust_claude: bool = True) -> list[dict]:
        """Index entries: with `success` set, only those whose outcome (Claude's label when there is one, it is not
        low-confidence and `trust_claude`, else the recorded one) matches."""
        out = []
        for e in self.entries():
            if sources and e.get('source') not in sources:
                continue
            use_claude = trust_claude and 'claude_success' in e and e.get('claude_confidence') != 'low'
            ok = e.get('claude_success') if use_claude else e.get('success')
            if success is not None and bool(ok) != success:
                continue
            out.append(e)
        return out


class EpisodeRecorder:
    """Records local-skill attempts during agent runs (core/skill_contact.py calls begin / step / end)."""

    def __init__(self, root: pathlib.Path | None = None, source: str = 'agent'):
        self.root, self.source = root, source
        self.ep: Episode | None = None

    def begin(self, skill: str, meta: dict) -> None:
        self.ep = Episode(skill, meta=dict(meta))

    def step(self, obs: dict, action, expert=None, intervened: bool = False, reward: float = 0.0) -> None:
        if self.ep is not None:
            self.ep.add(obs, action, expert, intervened, reward)

    def end(self, success: bool, info: dict | None = None) -> str:
        if self.ep is None:
            return ''
        eid = EpisodeStore(self.ep.skill, self.root).save(self.ep, source=self.source, success=bool(success),
                                                           judged_by='check', **(info or {}))
        self.ep = None
        return eid
