"""Build every hardware config from one joint table.

Inputs:
  * config/actuator_models.yaml  per-model numbers
  * config/joints.yaml           joint -> bus position, model, direction
  * calibration file             per-joint zero offsets
  * the URDF                     joint names and position limits

Outputs (written by the launch file at start-up):
  * one EtherCAT slave config per drive, with the joint's direction and zero
    offset folded into the PDO scale factors
  * a hardware manifest that the xacro reads to build the ros2_control block
  * parameters for the command controller (limits per joint)
"""

import copy
import math
import os
import xml.etree.ElementTree as ET

import yaml

CONTROL_MODES = {
    # control_mode: (CiA402 mode of operation at start-up, ros2_control command interface)
    'velocity': (9, 'velocity'),
    'effort': (10, 'effort'),
    # The arm stack's Mode P / Mode T (spec: Hardware assumptions): starts in
    # CSP (8); the ArmController switches 0x6060 to CST (10) at runtime. Each
    # joint exports position (0x607A), effort (0x6071), torque_offset (0x60B2,
    # gravity + friction feedforward in CSP), mode_of_operation (0x6060) and
    # max_torque (0x6072, the command's torque clamp, which caps the drive's
    # own CSP loop too).
    'arm': (8, 'arm'),
}
ARM_COMMAND_INTERFACES = ['position', 'effort', 'torque_offset', 'mode_of_operation', 'max_torque']

MOVABLE_JOINT_TYPES = ('revolute', 'continuous', 'prismatic')


class ConfigError(ValueError):
    """Raised when the joint table, model table or calibration is inconsistent."""


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def read_urdf_joints(urdf_xml):
    """Return {joint_name: (lower, upper)} for movable joints, in URDF order."""
    root = ET.fromstring(urdf_xml)
    joints = {}
    for joint in root.findall('joint'):
        if joint.get('type') not in MOVABLE_JOINT_TYPES:
            continue
        if joint.find('mimic') is not None:
            continue    # computed by robot_state_publisher, never a drive
        limit = joint.find('limit')
        lower = -math.inf
        upper = math.inf
        if limit is not None and joint.get('type') != 'continuous':
            lower = float(limit.get('lower', -math.inf))
            upper = float(limit.get('upper', math.inf))
        joints[joint.get('name')] = (lower, upper)
    return joints


class HardwareJoint:
    """One drive on the ring and the joint it moves."""

    def __init__(self, row, model, calibration, urdf_limits):
        self.name = row['name']
        self.bus_position = int(row['bus_position'])
        self.model_name = row['model']
        self.pdo_profile = row.get('pdo_profile', '26042402')
        if self.pdo_profile not in ('legacy', '26042402'):
            raise ConfigError('joint %r: pdo_profile must be legacy or 26042402' % self.name)
        self.supports_torque_offset = self.pdo_profile != 'legacy'
        self.direction = int(row.get('direction', 1))
        self.model = model
        self.zero_offset = calibration.get(self.name)
        self.calibrated = self.zero_offset is not None
        if self.zero_offset is None:
            self.zero_offset = 0.0
        self.zero_offset = float(self.zero_offset)

        self.max_velocity = float(
            row.get('max_velocity_rad_s', model['bringup_velocity_limit_rad_s']))
        self.max_effort = float(row.get('max_effort_nm', model['bringup_effort_limit_nm']))
        self.max_torque_permille = int(row.get('max_torque_permille', model['max_torque_permille']))
        self.damping = float(row.get('damping_nm_s_per_rad', model['damping_nm_s_per_rad']))
        self.k_stiction = float(row.get('k_stiction', model.get('k_stiction', 0.0)))

        if 'soft_limits_rad' in row:
            lower, upper = row['soft_limits_rad']
            self.position_limits = (float(lower), float(upper))
        else:
            self.position_limits = urdf_limits

    # Scale factors. The driver applies  value = factor * raw + offset  when
    # reading and  raw = factor * value + offset  when writing.
    @property
    def rad_per_count(self):
        return 2.0 * math.pi / float(self.model['counts_per_rev'])

    def position_state(self):
        return self.direction * self.rad_per_count, -self.zero_offset

    def position_command(self):
        # Inverse of position_state(), so the driver's "hold actual position"
        # default for 0x607A writes counts, not radians.
        factor = self.direction / self.rad_per_count
        return factor, factor * self.zero_offset

    def velocity_state(self):
        return self.direction * self.rad_per_count

    def velocity_command(self):
        return self.direction / self.rad_per_count

    def effort_state(self):
        return self.direction * float(self.model['effort_nm_per_permille'])

    def effort_command(self):
        return self.direction / float(self.model['effort_nm_per_permille'])


class HardwareConfig:
    """Validated joint table, ready to generate configs from."""

    def __init__(self, models_file, joints_file, urdf_xml, calibration_file=None,
                 control_mode=None, template_file=None):
        models = load_yaml(models_file)
        table = load_yaml(joints_file)
        calibration = load_yaml(calibration_file) if calibration_file else {}
        if calibration.get('coordinate_frame', 'urdf') not in ('urdf', 'relative'):
            raise ConfigError('Unknown calibration coordinate_frame')
        self.calibration = calibration
        self.auto_state_transitions = True

        self.defaults = models.get('defaults', {})
        self.models = models.get('models', {})
        self.template = load_yaml(template_file) if template_file else None
        self.calibration_version = calibration.get('version')
        self.kt_calibration = calibration.get('kt_output_nm_per_a') or {}
        zero_offsets = calibration.get('zero_offsets_rad') or {}

        self.control_mode = control_mode or table.get('control_mode', 'velocity')
        if self.control_mode not in CONTROL_MODES:
            raise ConfigError(
                'control_mode must be one of %s, got %r' % (sorted(CONTROL_MODES),
                                                            self.control_mode))
        self.mode_of_operation, self.command_interface = CONTROL_MODES[self.control_mode]
        self.command_interfaces = (list(ARM_COMMAND_INTERFACES) if self.control_mode == 'arm'
                                   else [self.command_interface])

        self.urdf_joints = read_urdf_joints(urdf_xml)
        rows = table.get('joints') or []
        self.joints = self._build_joints(rows, zero_offsets)
        # stage 3 (`arm limits`) writes soft limits into the calibration file;
        # they win over the URDF unless the joint table overrides them
        soft = calibration.get('soft_limits_rad') or {}
        for joint, row in zip(sorted(self.joints, key=lambda j: j.bus_position),
                              sorted(rows, key=lambda r: int(r['bus_position']))):
            lo, hi = (soft.get(joint.name) or [None, None])
            if 'soft_limits_rad' not in row and lo is not None and hi is not None:
                joint.position_limits = (float(lo), float(hi))
        if calibration.get('coordinate_frame') == 'relative':
            if self.control_mode != 'arm':
                raise ConfigError('A relative reference supports arm position holding only.')
            # CAD limits do not describe an arbitrary encoder reference. The
            # controller rejects motion commands and holds its startup position.
            local_motion = (calibration.get('commissioning') or {}).get('relative_motion', False)
            for joint in self.joints:
                if local_motion:
                    lo, hi = (soft.get(joint.name) or [None, None])
                    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi or hi-lo > math.radians(20) + 1e-12:
                        raise ConfigError('Relative motion needs finite local bounds no wider than 20 degrees.')
                    joint.position_limits = (float(lo), float(hi))
                else:
                    joint.position_limits = (-math.inf, math.inf)
        hw_names = {j.name for j in self.joints}
        self.mock_joints = [n for n in self.urdf_joints if n not in hw_names]

        unknown_cal = sorted(set(zero_offsets) - set(self.urdf_joints))
        if unknown_cal:
            raise ConfigError('calibration has zero offsets for joints not in the URDF: %s'
                              % ', '.join(unknown_cal))

    def _build_joints(self, rows, zero_offsets):
        joints = []
        seen_names = set()
        seen_positions = set()
        for i, row in enumerate(rows):
            for key in ('name', 'bus_position', 'model'):
                if key not in row:
                    raise ConfigError('joints[%d] is missing %r' % (i, key))
            name = row['name']
            if name not in self.urdf_joints:
                raise ConfigError('joint %r is not a movable joint in the URDF' % name)
            if name in seen_names:
                raise ConfigError('joint %r is listed twice' % name)
            if row['model'] not in self.models:
                raise ConfigError('joint %r uses unknown model %r (known: %s)'
                                  % (name, row['model'], ', '.join(sorted(self.models))))
            if int(row.get('direction', 1)) not in (1, -1):
                raise ConfigError('joint %r: direction must be 1 or -1' % name)
            position = int(row['bus_position'])
            if position in seen_positions:
                raise ConfigError('bus_position %d is used by more than one joint' % position)
            seen_names.add(name)
            seen_positions.add(position)
            model = dict(self.models[row['model']])
            model['counts_per_rev'] = row.get('counts_per_rev') or model['counts_per_rev']
            kt = self.kt_calibration.get(name)
            if kt is not None:
                rated = row.get('rated_current_a') or model['rated_current_a']
                if not math.isfinite(float(kt)) or float(kt) <= 0:
                    raise ConfigError('joint %r: calibrated torque constant must be positive' % name)
                model['effort_nm_per_permille'] = float(kt) * float(rated) / 1000
            joints.append(HardwareJoint(row, model,
                                        zero_offsets, self.urdf_joints[name]))

        expected = set(range(len(joints)))
        if seen_positions != expected:
            raise ConfigError(
                'bus positions must be 0..%d with no gaps, got %s'
                % (len(joints) - 1, sorted(seen_positions)))
        joints.sort(key=lambda j: j.bus_position)
        return joints

    def warnings(self):
        """Problems worth printing at launch that do not stop it."""
        out = []
        for name, model in sorted(self.models.items()):
            rated = 1000.0 * float(model['effort_nm_per_permille'])
            if rated > float(model['peak_effort_nm']):
                out.append(
                    'model %s: effort_nm_per_permille implies %.1f Nm at rated current, above '
                    'its %.1f Nm peak; effort readings and commands are suspect until Kt is '
                    'measured' % (name, rated, float(model['peak_effort_nm'])))
        uncalibrated = [j.name for j in self.joints if not j.calibrated]
        if uncalibrated:
            out.append('no zero offset for %s: position limits assume the encoder zero is the '
                       'URDF zero' % ', '.join(uncalibrated))
        return out

    # ------------------------------------------------------------------ outputs
    def slave_config(self, joint):
        """EtherCAT slave config dict for one drive."""
        if self.template is None:
            raise ConfigError('no PDO template loaded')
        cfg = copy.deepcopy(self.template)
        cfg['vendor_id'] = int(self.defaults.get('vendor_id', 0))
        cfg['product_id'] = int(self.defaults.get('product_id', 0))
        cfg['auto_state_transitions'] = self.auto_state_transitions
        dc = int(self.defaults.get('dc_assign_activate', 0))
        if dc and joint.pdo_profile != 'legacy':
            cfg['assign_activate'] = dc
        cfg['sdo'] = [{'index': 0x6060, 'sub_index': 0, 'type': 'int8',
                       'value': self.mode_of_operation}]

        pos_factor, pos_offset = joint.position_state()
        cmd_pos_factor, cmd_pos_offset = joint.position_command()
        arm = self.control_mode == 'arm'
        if joint.pdo_profile == 'legacy':
            # Firmware 5.12 has fixed base PDOs, with 0x5FFE padding in both
            # directions and no 0x1603 / 0x60B2 torque-offset objects.
            for pdo in cfg['rpdo'] + cfg['tpdo']:
                for channel in pdo['channels']:
                    if channel['index'] in (0x2ffd, 0x2ffe):
                        channel['index'] = 0x5ffe
        if arm and joint.supports_torque_offset:
            # 0x60B2 torque offset (CSP feedforward). The drive carries it in its
            # optional RxPDO 0x1603 (0x60B0, 0x60B1, 0x60B2: the vendor's CSP
            # mapping, firmware 26042402+), assigned next to 0x1600, both at their
            # default contents so no remapping is needed.
            cfg['rpdo'].append({'index': 0x1603, 'channels': [
                {'index': 0x60b0, 'sub_index': 0, 'type': 'int32', 'default': 0},   # position offset
                {'index': 0x60b1, 'sub_index': 0, 'type': 'int32', 'default': 0},   # velocity offset
                {'index': 0x60b2, 'sub_index': 0, 'type': 'int16', 'default': 0},   # torque offset
            ]})
        for pdo in cfg['rpdo']:
            for ch in pdo['channels']:
                index = ch['index']
                if index == 0x607a:
                    ch['factor'] = cmd_pos_factor
                    ch['offset'] = cmd_pos_offset
                    if arm:
                        ch['command_interface'] = 'position'
                elif index == 0x60b2:
                    ch['factor'] = joint.effort_command()
                    ch['command_interface'] = 'torque_offset'
                elif index == 0x60ff:
                    ch['factor'] = joint.velocity_command()
                    if self.command_interface == 'velocity':
                        ch['command_interface'] = 'velocity'
                elif index == 0x6071:
                    ch['factor'] = joint.effort_command()
                    if self.command_interface == 'effort' or arm:
                        ch['command_interface'] = 'effort'
                elif index == 0x6072:
                    ch['default'] = joint.max_torque_permille
                    if arm:
                        # the ArmController writes the command's torque clamp here
                        # a magnitude: no direction sign
                        ch['factor'] = abs(joint.effort_command())
                        ch['command_interface'] = 'max_torque'
                elif index == 0x6060:
                    ch['default'] = self.mode_of_operation
                    if arm:
                        ch['command_interface'] = 'mode_of_operation'
        for pdo in cfg['tpdo']:
            for ch in pdo['channels']:
                index = ch['index']
                if index == 0x6064:
                    ch['state_interface'] = 'position'
                    ch['factor'] = pos_factor
                    ch['offset'] = pos_offset
                elif index == 0x606c:
                    ch['state_interface'] = 'velocity'
                    ch['factor'] = joint.velocity_state()
                elif index == 0x6077:
                    ch['state_interface'] = 'effort'
                    ch['factor'] = joint.effort_state()
        return cfg

    def write(self, out_dir, use_mock_hardware=False, control_frequency=100.0):
        """Write slave configs and the hardware manifest; return the manifest path."""
        os.makedirs(out_dir, exist_ok=True)
        ethercat = []
        for joint in self.joints:
            path = os.path.join(out_dir, 'drive_%02d_%s.yaml' % (joint.bus_position, joint.name))
            with open(path, 'w') as f:
                yaml.safe_dump(self.slave_config(joint), f, sort_keys=False)
            ethercat.append({
                'name': joint.name,
                'position': joint.bus_position,
                'slave_config': path,
                'command_interfaces': [name for name in self.command_interfaces
                                       if name != 'torque_offset' or joint.supports_torque_offset],
            })
        manifest = {
            'use_mock_hardware': bool(use_mock_hardware),
            'control_frequency': float(control_frequency),
            'mode_of_operation': self.mode_of_operation,
            'command_interface': self.command_interface,
            'command_interfaces': list(self.command_interfaces),
            'ethercat': ethercat,
            'mock': list(self.mock_joints),
        }
        manifest_path = os.path.join(out_dir, 'hardware_manifest.yaml')
        with open(manifest_path, 'w') as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        return manifest_path

    def controller_name(self):
        return '%s_controller' % self.command_interface

    def arm_controller_parameters(self, arm_config=None, calibration_file=None):
        """ros__parameters for the ArmController (control_mode: arm)."""
        arm_config = arm_config or {}
        rows = {r['name']: r for r in arm_config.get('joints') or []}
        gains = [rows.get(j.name, {}).get('gains') or {} for j in self.joints]
        relative = self.calibration.get('coordinate_frame') == 'relative'
        local_motion = relative and (self.calibration.get('commissioning') or {}).get('relative_motion', False)
        params = {
            'joints': [j.name for j in self.joints],
            'max_torque': [float(rows.get(j.name, {}).get('max_torque_nm') or j.max_effort)
                           for j in self.joints],
            'max_velocity': [float(rows.get(j.name, {}).get('max_vel_rad_s') or j.max_velocity)
                             for j in self.joints],
            'soft_lower': [_finite(j.position_limits[0], -1e9) for j in self.joints],
            'soft_upper': [_finite(j.position_limits[1], 1e9) for j in self.joints],
            'kp': [float(g.get('kp', 0.0)) for g in gains],
            'kd': [float(g.get('kd', 0.0)) for g in gains],
            'kd_fallback': [max(float(g.get('kd', 0.0)), j.damping)
                            for g, j in zip(gains, self.joints)],
            'watchdog_s': 0.05,
            'write_max_torque': True,
            'drive_feedback': True,
            'hold_on_startup': True,
            'hold_on_timeout': True,
            'hold_only': relative and not local_motion,
            'position_only': relative,
            'gravity_enabled': self.calibration.get('coordinate_frame') != 'relative' and
                bool(self.calibration.get('link_inertials')) and
                (self.calibration.get('gravity_validation') or {'passed': True}).get('passed', False),
            'torque_offset_enabled': [j.supports_torque_offset for j in self.joints],
            'calibration_file': calibration_file or '',
        }
        local = self.calibration.get('local_gravity')
        if relative and local_motion and local:
            n = len(self.joints)
            if local.get('kind') != 'local_affine' or local.get('validated') is not True:
                raise ConfigError('Local gravity must have passed independent validation.')
            fields = {'local_center': 'center_rad', 'local_bias': 'bias_nm',
                      'local_lower': 'lower_rad', 'local_upper': 'upper_rad'}
            for param, key in fields.items():
                values = local.get(key) or []
                if len(values) != n or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                    raise ConfigError('Invalid local gravity ' + key)
                params[param] = [float(v) for v in values]
            matrix = local.get('jacobian_nm_per_rad') or []
            if len(matrix) != n or any(not isinstance(row, list) or len(row) != n for row in matrix):
                raise ConfigError('Invalid local gravity Jacobian dimensions')
            flat = [v for row in matrix for v in row]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in flat):
                raise ConfigError('Invalid local gravity Jacobian entries')
            params['local_jacobian'] = [float(v) for v in flat]
        # Weighed link masses: the Python model (sim, calibration) applies them
        # under the calibration's link_inertials, so the controller's gravity
        # model must too. Nulls (not weighed yet) keep the URDF mass; ROS
        # parameter files cannot hold an empty list, so omit the pair then.
        masses = {k: float(v) for k, v in (arm_config.get('link_masses_kg') or {}).items()
                  if v is not None}
        if masses:
            params['link_mass_names'] = list(masses)
            params['link_mass_values'] = list(masses.values())
        return params

    def controller_parameters(self, command_timeout=0.1, soft_margin=0.1):
        """ros__parameters for the SafeCommandController."""
        if self.command_interface == 'velocity':
            max_command = [j.max_velocity for j in self.joints]
        else:
            max_command = [j.max_effort for j in self.joints]
        return {
            'joints': [j.name for j in self.joints],
            'interface_name': self.command_interface,
            'command_timeout': float(command_timeout),
            'soft_margin': float(soft_margin),
            'max_command': max_command,
            'position_min': [_finite(j.position_limits[0], -1e9) for j in self.joints],
            'position_max': [_finite(j.position_limits[1], 1e9) for j in self.joints],
            'damping': [j.damping for j in self.joints],
        }

    def controllers_yaml(self, update_rate, joint_state_rate, command_timeout=0.1,
                         arm_config=None, calibration_file=None):
        """Full parameter file for ros2_control_node and the spawners."""
        name = self.controller_name()
        if self.control_mode == 'arm':
            return {
                'controller_manager': {'ros__parameters': {
                    'update_rate': int(update_rate),
                    'joint_state_broadcaster': {
                        'type': 'joint_state_broadcaster/JointStateBroadcaster'},
                    name: {'type': 'wisco_ethercat_ros2_hw/ArmController'},
                }},
                'joint_state_broadcaster': {'ros__parameters': {
                    'update_rate': int(min(joint_state_rate, update_rate))}},
                name: {'ros__parameters': self.arm_controller_parameters(arm_config,
                                                                         calibration_file)},
            }
        return {
            'controller_manager': {'ros__parameters': {
                'update_rate': int(update_rate),
                'joint_state_broadcaster': {
                    'type': 'joint_state_broadcaster/JointStateBroadcaster'},
                name: {'type': 'wisco_ethercat_ros2_hw/SafeCommandController'},
            }},
            'joint_state_broadcaster': {'ros__parameters': {
                'update_rate': int(min(joint_state_rate, update_rate))}},
            name: {'ros__parameters': self.controller_parameters(command_timeout)},
        }


def _finite(value, fallback):
    return value if math.isfinite(value) else fallback


def load_joint_profiles(models_file, joints_file):
    """{joint: {'model', 'bus_position', 'direction', 'k_stiction'}} from the tables alone.

    For scripts that need per-joint model data but not the URDF.
    """
    models = load_yaml(models_file).get('models', {})
    out = {}
    for row in load_yaml(joints_file).get('joints') or []:
        model = models.get(row['model'], {})
        out[row['name']] = {
            'model': row['model'],
            'bus_position': int(row['bus_position']),
            'direction': int(row.get('direction', 1)),
            'k_stiction': float(row.get('k_stiction', model.get('k_stiction', 0.0))),
        }
    return out
