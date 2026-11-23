"""Claude as the labeller: was this attempt a success? For filtering training data and for the success classifier's
labels, never as the RL reward itself (a judge used as a reward gets exploited: it fires on "near the object" or
the wrong object; research notes, "learning on the job").

    .venv/bin/python -m learning.label --skill insert --limit 20        # label episodes not labelled yet
    .venv/bin/python -m learning.label --skill insert --check            # agreement with the recorded outcome

For each episode Claude sees the policy's first wrist frame and full-resolution wrist and overhead images of how it
ended (stored with the episode; older episodes without them: first, middle and last 64-pixel frames, which turned
out too small to judge an insertion by), and the skill's success question,
and answers in a fixed JSON form (structured output): success, confidence, and a one-line reason. The answer is
appended to the episode index (`claude_success`, `claude_confidence`, `claude_reason`), where `EpisodeStore.filter`
and the classifier training read it. On the sim, `--check` compares Claude's labels with the simulator's truth: the
number to look at before trusting the labels on the rig.

Model: PANTHERA_LABEL_MODEL (default claude-opus-5-5), key ANTHROPIC_API_KEY from the repo's .env. Each label is one
request with three small images.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np

from learning.episodes import EpisodeStore

QUESTIONS = {
    'insert': 'The robot is putting a peg it holds into a hole in a block. Judge only the last frame against the '
              'others: is the peg now inside the hole (gone down into it, not resting on top of the block or beside '
              'it)?',
    'pick': 'Is the object now held between the gripper\'s fingers, lifted off the table?',
    'place': 'Is the object now resting where it was meant to go, released by the gripper?',
    'pour': 'Did the beads end up in the bowl?',
}

SCHEMA = {'type': 'object', 'properties': {
    'success': {'type': 'boolean'},
    'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
    'reason': {'type': 'string'}}, 'required': ['success', 'confidence', 'reason'], 'additionalProperties': False}


def _jpeg(img: np.ndarray) -> str:
    big = img if min(img.shape[:2]) >= 128 else cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf.tobytes()).decode()


def label_frames(frames: list[tuple], question: str, client=None, model: str | None = None) -> dict:
    """Claude's verdict on an attempt from a few (caption, image, longest side in pixels) frames."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    content = []
    for name, f, size in frames:
        h, w = f.shape[:2]
        img = f if max(h, w) <= size else cv2.resize(f, (size, int(size * h / w)), interpolation=cv2.INTER_AREA)
        content += [{'type': 'text', 'text': f'[{name}]'},
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': _jpeg(img)}}]
    content.append({'type': 'text', 'text': question + ' Answer from what the images show; if they do not show '
                    'it clearly, say so with low confidence.'})
    resp = client.messages.create(model=model or os.environ.get('PANTHERA_LABEL_MODEL') or 'claude-opus-5-5',
                                  max_tokens=1024, messages=[{'role': 'user', 'content': content}],
                                  output_config={'format': {'type': 'json_schema', 'schema': SCHEMA}})
    text = next(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
    return json.loads(text)


def label_store(skill: str, limit: int = 20, client=None, log=print) -> list[dict]:
    store = EpisodeStore(skill)
    todo = [e for e in store.entries() if 'claude_success' not in e and e.get('source') != 'eval'][:limit]
    out = []
    for e in todo:
        z = store.load(e['id'])
        w = z['wrist']
        views = store.views(e['id'])
        if views:      # the start (policy frame) and how it ended, at full resolution
            frames = [('first', w[0], 256)] + [(f'last, {k} camera', v, 768) for k, v in sorted(views.items())]
        else:
            frames = [('first', w[0], 256), ('middle', w[len(w) // 2], 256), ('last', w[-1], 256)]
        v = label_frames(frames, QUESTIONS.get(skill, f'Did the {skill} attempt succeed?'), client)
        store.label(e['id'], claude_success=bool(v['success']), claude_confidence=v['confidence'],
                    claude_reason=v['reason'])
        row = dict(id=e['id'], recorded=e.get('success'), claude=v['success'], confidence=v['confidence'],
                   reason=v['reason'])
        log(row)
        out.append(row)
    return out


def agreement(skill: str) -> dict:
    """How often Claude's labels match the recorded outcome (the simulator's truth on the sim)."""
    rows = [e for e in EpisodeStore(skill).entries() if 'claude_success' in e and 'success' in e]
    if not rows:
        return dict(n=0)
    agree = [bool(e['claude_success']) == bool(e['success']) for e in rows]
    fp = [bool(e['claude_success']) and not e['success'] for e in rows]
    return dict(n=len(rows), agreement=round(float(np.mean(agree)), 3),
                false_positives=int(sum(fp)), false_negatives=int(sum(not a and not f for a, f in zip(agree, fp))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skill', default='insert')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--check', action='store_true', help='only report agreement with the recorded outcomes')
    a = ap.parse_args()
    if not a.check:
        from core.build import load_dotenv
        load_dotenv()
        label_store(a.skill, a.limit, log=lambda d: print(json.dumps(d), flush=True))
    print(json.dumps(agreement(a.skill)))


if __name__ == '__main__':
    main()
