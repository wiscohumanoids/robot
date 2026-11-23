"""The tools the LLM can call, and their dispatch to the skills.

Learning: an object not seen before is learned by choosing numbered points on it (propose_points, then learn); the
LLM answers with numbers, never pixel coordinates.

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
    {'name': 'pick', 'description': 'Pick up a found object by its number: choose a grasp from its shape (or by a '
     'learned part, e.g. "handle" or "corner"), approach from above, refine with the wrist camera, grasp, lift, and '
     'check the grasp by the jaw width and by sight. The arm must be empty.',
     'input_schema': _schema({'number': {'type': 'integer'},
                              'part': {'type': ['string', 'null'], 'description': 'a learned part to grasp it by, '
                                       'or null for the best grasp of the whole object'}})},
    {'name': 'place_on', 'description': 'Set the held object on top of a found object (by number), faces aligned, '
     'release it, and measure where it ended up.',
     'input_schema': _schema({'number': {'type': 'integer'}})},
    {'name': 'place_at', 'description': 'Set the held object down on the table with its centre at (x, y) cm, read '
     'off the grid on the overhead image, and release it.',
     'input_schema': _schema({'x_cm': {'type': 'number'}, 'y_cm': {'type': 'number'}})},
    {'name': 'propose_points', 'description': 'Numbered points on the things on the table in a fresh image, for '
     'learning an object not seen before. Points in one region are on one thing; choose them by number in learn.',
     'input_schema': _schema({'camera': {'type': 'string', 'enum': ['overhead']},
                              'detail': {'type': ['string', 'null'], 'description': 'a region letter from the '
                                         'previous propose_points: dense points over that region, to mark a part '
                                         'precisely; null for all things on the table'}})},
    {'name': 'learn', 'description': 'Learn an object from the latest propose_points image: its name, the points on it '
     '(all of them, or the ones clearly on it), and optionally named parts (one point each: e.g. handle, rim, hook, '
     'corner, hole). The arm then looks at it from the wrist camera to measure its height and shape. Afterwards '
     'find("<name>") finds it, and pick and the other skills use its shape and parts.',
     'input_schema': _schema({
         'name': {'type': 'string', 'description': 'a short name with its colour, e.g. "purple mug"'},
         'points': {'type': 'array', 'items': {'type': 'integer'}},
         'parts': {'type': 'array', 'items': _schema({'name': {'type': 'string'}, 'point': {'type': 'integer'}})},
         'deformable': {'type': 'boolean', 'description': 'soft or floppy (cloth, a bag, a cable)'}})},
    {'name': 'known_objects', 'description': 'The objects learned so far, with their measured shapes and parts.',
     'input_schema': _schema({})},
    {'name': 'insert', 'description': 'Insert the held object into the hole of a found object (by number): line up '
     'above it, lower to the opening, and let the local insertion skill finish (it presses and searches; a trained '
     'policy when there is one). Fails, still holding, if it does not go in.',
     'input_schema': _schema({'number': {'type': 'integer'}})},
    {'name': 'pull_with_tool', 'description': 'With a held tool that has a learned "hook" part, reach past a found '
     'object (by number) and drag it toward the robot, into reach.',
     'input_schema': _schema({'number': {'type': 'integer'}})},
    {'name': 'pour_into', 'description': 'Tip the held container over a found object (by number), hold, and tip it '
     'back.',
     'input_schema': _schema({'number': {'type': 'integer'}})},
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
        return skills.pick(int(args['number']), args.get('part') or None)
    if name == 'propose_points':
        return skills.propose_points(args.get('camera') or 'overhead', args.get('detail') or None)
    if name == 'learn':
        parts = {p['name']: int(p['point']) for p in args.get('parts') or []}
        return skills.learn(args['name'], [int(n) for n in args['points']], parts, bool(args.get('deformable')))
    if name == 'known_objects':
        return skills.known_objects()
    if name == 'insert':
        return skills.insert(int(args['number']))
    if name == 'pull_with_tool':
        return skills.pull_with_tool(int(args['number']))
    if name == 'pour_into':
        return skills.pour_into(int(args['number']))
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
