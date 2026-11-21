"""Run physical-frame inference in the DINO environment over a local pipe."""

import base64
import json
import os
import sys


def process_frame(vision, item):
    """Decode a captured image pair and dispatch its encoder measurements with it."""
    import cv2
    import numpy as np

    op = item.get('op', 'locate')
    if op not in ('calibrate', 'finish', 'locate'):
        raise ValueError(f'Unknown vision operation {op}')

    def decode(name):
        compressed = np.frombuffer(base64.b64decode(item[name], validate=True), dtype=np.uint8)
        image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not decode {name} camera JPEG')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overhead, wrist = decode('overhead'), decode('wrist')
    if op == 'locate':
        return vision.locate(overhead, wrist, item['q'])
    return vision.calibrate(overhead, wrist, item['q'], item['gripper'], finish=op == 'finish')


def serve():
    # Some model loaders write progress to stdout, including through native code.
    output = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    vision = None
    try:
        for line in sys.stdin:
            try:
                item = json.loads(line)
                if vision is None:
                    config = item['config']
                    if config.get('auto_calibrate') and sys.platform.startswith('linux') and not os.getenv('DISPLAY'):
                        os.environ.setdefault('MUJOCO_GL', 'egl')
                    from stack_vision import StackVision
                    vision = StackVision(config)
                    result = True
                else:
                    result = process_frame(vision, item)
                output.write(json.dumps({'result': result}, allow_nan=False) + '\n')
            except Exception as exc:
                output.write(json.dumps({'error': str(exc)}) + '\n')
    finally:
        if vision is not None:
            vision.close()
        output.close()


if __name__ == '__main__':
    serve()
