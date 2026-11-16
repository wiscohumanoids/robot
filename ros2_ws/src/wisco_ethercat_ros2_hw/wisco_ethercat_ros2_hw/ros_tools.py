"""Small rclpy helpers shared by the scripts."""

import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

_FIELDS = {
    ParameterType.PARAMETER_BOOL: 'bool_value',
    ParameterType.PARAMETER_INTEGER: 'integer_value',
    ParameterType.PARAMETER_DOUBLE: 'double_value',
    ParameterType.PARAMETER_STRING: 'string_value',
    ParameterType.PARAMETER_BYTE_ARRAY: 'byte_array_value',
    ParameterType.PARAMETER_BOOL_ARRAY: 'bool_array_value',
    ParameterType.PARAMETER_INTEGER_ARRAY: 'integer_array_value',
    ParameterType.PARAMETER_DOUBLE_ARRAY: 'double_array_value',
    ParameterType.PARAMETER_STRING_ARRAY: 'string_array_value',
}

LIMIT_PARAMETERS = ['joints', 'interface_name', 'max_command', 'position_min', 'position_max']


def get_remote_parameters(node, target_node, names, timeout_sec=30.0):
    """Read parameters from another node; raise RuntimeError if it is not there."""
    client = node.create_client(GetParameters, '/%s/get_parameters' % target_node.lstrip('/'))
    try:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError('%s is not running (no /%s/get_parameters service)'
                               % (target_node, target_node))
        future = client.call_async(GetParameters.Request(names=list(names)))
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if future.result() is None:
            raise RuntimeError('no reply from %s/get_parameters' % target_node)
        out = {}
        for name, value in zip(names, future.result().values):
            field = _FIELDS.get(value.type)
            if field is None:
                raise RuntimeError('%s has no parameter %r' % (target_node, name))
            v = getattr(value, field)
            out[name] = list(v) if field.endswith('array_value') else v
        return out
    finally:
        node.destroy_client(client)


def get_controller_limits(node, controller, timeout_sec=30.0):
    """The joint list and limits the SafeCommandController is running with.

    Reading them from the controller (rather than from the config files) means
    a script always agrees with what the controller will actually enforce.
    """
    return get_remote_parameters(node, controller, LIMIT_PARAMETERS, timeout_sec)


def wait_for_joint_states(node, joints, timeout_sec=10.0):
    """Block until /joint_states has reported every joint; return {name: (pos, vel)}."""
    from sensor_msgs.msg import JointState

    latest = {}

    def callback(msg):
        for i, name in enumerate(msg.name):
            pos = msg.position[i] if i < len(msg.position) else float('nan')
            vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
            latest[name] = (pos, vel)

    sub = node.create_subscription(JointState, '/joint_states', callback, 10)
    try:
        deadline = node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while rclpy.ok() and not all(j in latest for j in joints):
            if node.get_clock().now().nanoseconds > deadline:
                missing = [j for j in joints if j not in latest]
                raise RuntimeError('no joint_states for %s' % ', '.join(missing))
            rclpy.spin_once(node, timeout_sec=0.1)
        return dict(latest)
    finally:
        node.destroy_subscription(sub)
