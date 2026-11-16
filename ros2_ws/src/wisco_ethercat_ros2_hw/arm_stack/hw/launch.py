"""Own the ROS launch used by arm calibration and cyclic telemetry."""

from contextlib import closing, contextmanager
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import yaml


class ArmLaunch:
    def __init__(self, arm, calibration_file, log_path, *, enable_motion=True):
        self.stream = open(log_path, 'w')
        try:
            command = [
                'ros2', 'launch', 'wisco_ethercat_ros2_hw', 'arm01.launch.py',
                'joints_file:=' + arm.path, 'enable_motion:=' + str(enable_motion).lower(),
            ]
            if calibration_file:
                command.append('calibration_file:=' + str(calibration_file))
            self.process = subprocess.Popen(command, stdout=self.stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        except OSError as error:
            self.stream.close()
            raise RuntimeError('Unable to start the ROS launch: %s' % error) from error
        self.log_path = Path(log_path)

    def check_running(self):
        if self.process.poll() is not None:
            raise RuntimeError('Arm launch exited.\n' + self.log_path.read_text()[-4000:])

    def connect(self, arm, seed):
        from .ros_client import RosArmClient
        self.check_running()
        # The seven drives can take 6–8 s each to enter OP before controllers spawn.
        try:
            return RosArmClient(arm, seed, state_timeout=120.0, check_launch=self.check_running)
        except RuntimeError as error:
            if 'No /arm_controller/state' in str(error):
                raise RuntimeError(f'{error}\nLaunch output:\n{self.log_path.read_text()[-4000:]}') from error
            raise

    def close(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if self.process.poll() is not None:
                break
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                continue
        self.stream.close()


@contextmanager
def passive_feedback(arm, calibration):
    """Own one motor-disabled launch and pin its encoder conversion for the session."""
    with tempfile.TemporaryDirectory(prefix='arm_feedback_') as directory:
        snapshot = Path(directory) / 'calibration.yaml'
        snapshot.write_text(yaml.safe_dump(calibration))
        with closing(ArmLaunch(arm, snapshot, Path(directory) / 'launch.log', enable_motion=False)) as launch:
            with closing(launch.connect(arm, calibration)) as client:
                yield client
