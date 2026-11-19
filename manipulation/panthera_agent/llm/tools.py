"""The tools the LLM can call, and their dispatch to the skills.

Few and semantic on purpose: each one is a whole piece of work the geometry can do reliably (find and measure, pick,
place), so the model decides what to do next and never computes a coordinate from pixels. Positions it gives
(`place_at`) are read off the table grid drawn on the overhead image, in centimetres.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from core.skills import Result, Skills


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required,
            'additionalProperties': False}


TOOLS = [
    {'name': 'look', 'description': 'Take a fresh image from a camera. The overhead image has the table grid (x,y in '
     'cm). The wrist camera looks at the jaws. Also reports what the gripper holds.',
     'input_schema': _schema({'camera': {'type': 'string', 'enum': ['overhead', 'wrist']}})},
    {'name': 'find', 'description': 'Find every object matching a short description ("red cube", "block") in a fresh '
     'overhead image. Returns them numbered on the image with their measured position, size and yaw. Numbers are new '
     'each time and are what pick and place_on take. Positions are only valid until something moves: find again '
     'after any pick or place.',
     'input_schema': _schema({'query': {'type': 'string', 'description': 'object description, e.g. "green cube"'}})},
    {'name': 'pick', 'description': 'Pick up a found object by its number: approach from above, refine with the wrist '
     'camera, grasp, lift, and check the grasp from the jaw width. The arm must be empty.',
     'input_schema': _schema({'number': {'type': 'integer'}})},
    {'name': 'place_on', 'description': 'Set the held object on top of a found object (by number), faces aligned, '
     'and release it.',
     'input_schema': _schema({'number': {'type': 'integer'}})},
    {'name': 'place_at', 'description': 'Set the held object down on the table with its centre at (x, y) cm, read '
     'off the grid on the overhead image, and release it.',
     'input_schema': _schema({'x_cm': {'type': 'number'}, 'y_cm': {'type': 'number'}})},
    {'name': 'open_gripper', 'description': 'Open the jaws where the arm is (drops whatever is held). For recovery.',
     'input_schema': _schema({})},
    {'name': 'home', 'description': 'Move the empty arm back to its start pose, clear of the overhead camera\'s view.',
     'input_schema': _schema({})},
    {'name': 'done', 'description': 'Finish the task: whether the instruction is now satisfied, and a one-sentence '
     'summary. Check with find or look first.',
     'input_schema': _schema({'success': {'type': 'boolean'}, 'summary': {'type': 'string'}})},
]
for _t in TOOLS:
    _t['strict'] = True


def run_tool(skills: Skills, name: str, args: dict) -> Result:
    """Run one tool call."""
    if name == 'look':
        return skills.look(args['camera'])
    if name == 'find':
        return skills.find(args['query'])
    if name == 'pick':
        return skills.pick(int(args['number']))
    if name == 'place_on':
        return skills.place_on(int(args['number']))
    if name == 'place_at':
        return skills.place_at(float(args['x_cm']) / 100.0, float(args['y_cm']) / 100.0)
    if name == 'open_gripper':
        return skills.open_gripper()
    if name == 'home':
        return skills.home()
    return Result(False, f'Unknown tool {name!r}.')


def jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    """An RGB image as base64 JPEG."""
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return base64.standard_b64encode(buf.tobytes()).decode()


def content_blocks(result: Result) -> list[dict]:
    """A result as Messages API content: the message, then each image with its caption."""
    blocks = [{'type': 'text', 'text': ('' if result.ok else 'FAILED: ') + result.message}]
    for caption, img in result.images:
        blocks.append({'type': 'text', 'text': f'[{caption}]'})
        blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                   'data': jpeg_b64(img)}})
    return blocks
