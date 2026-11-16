"""Short SDK status probe; do not invoke SDK destructor motor commands."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import yaml
import hightorque_robot as htr


def finish(code):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

signal.signal(signal.SIGALRM, lambda *_: finish(124))
signal.alarm(15)
try:
    path = Path(sys.argv[1]).resolve()
    assert path.name in ('ttyACM0', 'ttyACM7'), path
    assert path.exists()
    template = Path(__file__).resolve().parents[1] / 'work/Panthera-HT_SDK/panthera_python/robot_param/motor_param/6dof_Panthera_params_follower.yaml'
    config = yaml.safe_load(template.read_text())
    config['robot']['Serial_Type'] = str(path)
    config['robot']['motor_timeout_ms'] = 0
    config_path = Path(__file__).resolve().parents[1] / 'logs' / ('probe-' + path.name + '.yaml')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    wrapper = Path(__file__).resolve().parents[1] / 'logs' / ('probe-' + path.name + '-robot.yaml')
    wrapper.write_text(yaml.safe_dump({'robot': {'name': 'Panthera-HT', 'robot_name': 'Panthera-HT', 'param_file': str(config_path.resolve())}}))
    robot = htr.Robot(str(wrapper.resolve()))
    motors = robot.get_motors()
    for _ in range(5):
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
        time.sleep(0.1)
    result = []
    for m in motors:
        s = m.get_current_motor_state()
        result.append(dict(id=m.get_motor_id(), position=s.position, velocity=s.velocity, torque=s.torque, mode=s.mode, fault=s.fault))
    print('PROBE_RESULT ' + json.dumps({'port': str(path), 'motors': result}))
    finish(0)
except BaseException:
    import traceback
    traceback.print_exc()
    finish(1)
