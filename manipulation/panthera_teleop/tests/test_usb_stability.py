from contextlib import redirect_stderr
import io
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import hand_guide
from usb_stability import is_serial_disconnect, keep_usb_awake, robot_usb_devices


class UsbStabilityTests(unittest.TestCase):
    def test_board_and_upstream_hubs_stay_awake(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub = root / 'devices/usb1/1-1'
            board = hub / '1-1.2'
            endpoint = board / '1-1.2:1.0/tty/ttyACM7'
            endpoint.mkdir(parents=True)
            ttys = root / 'class/tty'
            ttys.mkdir(parents=True)
            (ttys / 'ttyACM7').symlink_to(endpoint)
            for device in (hub, board):
                (device / 'idVendor').write_text('caf1')
                (device / 'idProduct').write_text('ffff')
                (device / 'power').mkdir()
                (device / 'power/control').write_text('auto\n')
            self.assertEqual(robot_usb_devices('/dev/ttyACM7', ttys), [board, hub])
            self.assertEqual(keep_usb_awake('/dev/ttyACM7', ttys), [])
            for device in (board, hub):
                self.assertEqual((device / 'power/control').read_text().strip(), 'on')

    def test_unprivileged_usb_setup_reports_actionable_instruction(self):
        control = Mock()
        control.exists.return_value = True
        control.read_text.return_value = 'auto'
        control.write_text.side_effect = PermissionError('root required')
        device = Mock()
        device.__truediv__ = Mock(return_value=control)
        with patch('usb_stability.robot_usb_devices', return_value=[device]), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(keep_usb_awake('/dev/ttyACM0'), [str(control)])
        self.assertIn('sudo ./scripts/enable-robot-usb.sh', output.getvalue())

    def test_disconnect_classifier_retains_real_control_faults(self):
        self.assertTrue(is_serial_disconnect(RuntimeError('left: IO Exception (9): Bad file descriptor')))
        self.assertTrue(is_serial_disconnect(RuntimeError('SerialException device disconnected?')))
        outer = RuntimeError('left: failed')
        outer.__cause__ = OSError(9, 'Bad file descriptor')
        self.assertTrue(is_serial_disconnect(outer))
        for message in ('motor fault', 'feedback stale for >0.5 seconds', 'Gripper command rejected'):
            self.assertFalse(is_serial_disconnect(RuntimeError(message)))

    def run_fault(self, fault):
        store = Mock(failure=None, control_requests=queue.Queue())
        dashboard = Mock()
        with patch.object(sys, 'argv', ['hand_guide.py', '--demo', '--no-browser']), \
                patch.object(hand_guide, 'EpisodeStore', return_value=store), \
                patch.object(hand_guide, 'Dashboard', return_value=dashboard), \
                patch.object(hand_guide, 'simulated_sample', side_effect=fault), \
                patch.object(hand_guide, 'stop_robots') as stop, \
                redirect_stderr(io.StringIO()):
            result = hand_guide.main()
        stop.assert_called_once_with({})
        store.close.assert_called_once_with('fault: ' + str(fault))
        dashboard.close.assert_called_once()
        return result

    def test_disconnect_finishes_session_without_uncaught_exception(self):
        self.assertEqual(self.run_fault(RuntimeError('IO Exception (9): Bad file descriptor')), 1)

    def test_non_transport_faults_still_raise(self):
        with self.assertRaisesRegex(RuntimeError, 'motor fault'):
            self.run_fault(RuntimeError('motor fault'))


if __name__ == '__main__':
    unittest.main()
