"""One blocking action call with a time limit (no rclpy, unit-testable).

This is the "succeeded within N s?" decision of the flow. `client` is an rclpy
ActionClient (or a fake with the same four methods); futures only need
done() / result() / add_done_callback().
"""
import time

from behavior_tree.skills import Outcome

SERVER_WAIT_S = 5.0
POLL_S = 0.05
# After a timeout the goal is cancelled, but the server only notices on its own next
# cycle and rejects new goals until then: wait this long for it to really finish.
CANCEL_WAIT_S = 5.0
STOPPED = 'stopped: behavior tree shutting down'

# action_msgs/msg/GoalStatus values
STATUS_SUCCEEDED = 4
STATUS_CANCELED = 5
STATUS_ABORTED = 6
_STATUS_NAMES = {STATUS_ABORTED: 'aborted', STATUS_CANCELED: 'canceled'}


def _never():
    return False


def _wait(future, deadline, clock, sleep, stop=_never):
    while not future.done():
        if stop() or clock() > deadline:
            return False
        sleep(POLL_S)
    return True


def _cancel_when_accepted(future):
    """A goal accepted after we gave up on it must not keep running."""
    handle = future.result()
    if handle is not None and handle.accepted:
        handle.cancel_goal_async()


def _stopped(handle=None):
    """Shutting down: cancel the goal (best effort, nothing is waited for) so the robot
    does not keep executing it, and tell the flow not to retry."""
    reason = STOPPED
    if handle is not None:
        try:
            handle.cancel_goal_async()
        except Exception as exc:  # the ROS context may already be gone
            reason += f' (cancel not sent: {exc})'
    return Outcome(False, reason, retryable=False)


def call_action(client, goal, timeout_s, manipulation=False,
                clock=time.monotonic, sleep=time.sleep, stop=_never):
    """Send `goal`, wait for the result for at most `timeout_s` in total, cancel
    the goal on timeout and wait up to CANCEL_WAIT_S more for the server to finish
    it (still a timeout, whatever the late result says). Returns an Outcome; never
    raises for action failures. `manipulation=True` also requires result.success.
    `stop()` turning True (node shutdown) cancels the goal and returns at once."""
    deadline = clock() + timeout_s
    if not client.wait_for_server(timeout_sec=min(SERVER_WAIT_S, timeout_s)):
        return Outcome(False, 'action server not available')

    send_future = client.send_goal_async(goal)
    if not _wait(send_future, deadline, clock, sleep, stop):
        send_future.add_done_callback(_cancel_when_accepted)
        return _stopped() if stop() else Outcome(False, f'timed out after {timeout_s:g} s')
    handle = send_future.result()
    if not handle.accepted:
        return Outcome(False, 'goal rejected')

    result_future = handle.get_result_async()
    if not _wait(result_future, deadline, clock, sleep, stop):
        if stop():
            return _stopped(handle)
        handle.cancel_goal_async()
        _wait(result_future, clock() + CANCEL_WAIT_S, clock, sleep, stop)
        return Outcome(False, f'timed out after {timeout_s:g} s')
    wrapped = result_future.result()
    if wrapped.status != STATUS_SUCCEEDED:
        name = _STATUS_NAMES.get(wrapped.status, f'status {wrapped.status}')
        return Outcome(False, f'action {name}')
    if manipulation and not wrapped.result.success:
        return Outcome(False, f'manipulation reported failure: {wrapped.result.message}')
    return Outcome(True)
