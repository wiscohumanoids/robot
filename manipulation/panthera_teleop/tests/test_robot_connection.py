from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import hand_guide
setup = hand_guide.setup


class RobotConnectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.board = root / 'devices/0000:06:00.3/usb1/1-1/1-1.2'
        self.board.mkdir(parents=True)
        self.ttys = root / 'class/tty'
        self.usb = root / 'bus/usb/devices'
        self.dev = root / 'dev'
        for directory in (self.ttys, self.usb, self.dev):
            directory.mkdir(parents=True)
        (self.usb / '1-1.2').symlink_to(self.board)
        (self.board / 'idVendor').write_text('caf1\n')
        (self.board / 'idProduct').write_text('ffff\n')
        (self.board / 'bConfigurationValue').write_text('1\n')
        self.mapping = dict(usb_path='1-1.2', pci_controller='0000:06:00.3')

    def resolve(self):
        return setup.resolve_board(self.mapping, self.ttys, self.usb, self.dev)

    def add_tty(self, interface='1-1.2:1.0'):
        endpoint = self.board / interface / 'tty/ttyACM0'
        endpoint.mkdir(parents=True)
        (self.ttys / 'ttyACM0').symlink_to(endpoint)
        (self.dev / 'ttyACM0').touch()

    def test_unconfigured_board_explains_power_and_absent_serial_port(self):
        for configuration in ('\n', '0\n'):
            (self.board / 'bConfigurationValue').write_text(configuration)
            with self.assertRaisesRegex(setup.RobotConnectionError, 'insufficient available bus power'):
                self.resolve()

    def test_configured_board_without_tty_explains_driver(self):
        with self.assertRaisesRegex(setup.RobotConnectionError, 'cdc_acm'):
            self.resolve()

    def test_disconnected_board_retains_original_port_instruction(self):
        (self.usb / '1-1.2').unlink()
        with self.assertRaisesRegex(setup.RobotConnectionError, 'original hub port'):
            self.resolve()

    def test_configured_serial_board_resolves_without_opening_it(self):
        self.add_tty()
        self.assertEqual(self.resolve(), self.dev / 'ttyACM0')

    def test_other_interface_does_not_replace_saved_control_interface(self):
        self.add_tty('1-1.2:1.2')
        with self.assertRaises(setup.RobotConnectionError):
            self.resolve()

    def test_wrong_usb_identity_is_rejected(self):
        (self.board / 'idVendor').write_text('1234')
        with self.assertRaisesRegex(setup.RobotConnectionError, 'Unexpected device'):
            self.resolve()

    def test_startup_preflight_failure_exits_cleanly_before_motor_or_camera_setup(self):
        with patch.object(sys, 'argv', ['hand_guide.py', '--run']), \
                patch.object(setup, 'prepare', side_effect=setup.RobotConnectionError('USB is unconfigured')), \
                patch.object(hand_guide, 'Cameras') as cameras, \
                patch.object(hand_guide, 'EpisodeStore') as store, \
                redirect_stderr(io.StringIO()) as output:
            self.assertEqual(hand_guide.main(), 1)
        cameras.assert_not_called()
        store.assert_not_called()
        self.assertIn('No motor devices opened', output.getvalue())
        self.assertNotIn('Traceback', output.getvalue())


if __name__ == '__main__':
    unittest.main()
