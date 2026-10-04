"""One blocking action call with a time limit (no rclpy, unit-testable).

This is the "succeeded within N s?" decision of the flow. `client` is an rclpy
ActionClient (or a fake with the same four methods); futures only need
done() / result() / add_done_callback().
"""
import time

from behavior_tree.skills import Outcome

SERVER_WAIT_S = 5.0
POLL_S = 0.05

# action_msgs/msg/GoalStatus values
STATUS_SUCCEEDED = 4
STATUS_CANCELED = 5
STATUS_ABORTED = 6
_STATUS_NAMES = {STATUS_ABORTED: 'aborted', STATUS_CANCELED: 'canceled'}


def _wait(future, deadline, clock, sleep):
    while not future.done():
        if clock() > deadline:
            return False
        sleep(POLL_S)
    return True


def _cancel_when_accepted(future):
    """A goal accepted after we gave up on it must not keep running."""
    handle = future.result()
    if handle is not None and handle.accepted:
        handle.cancel_goal_async()


def call_action(client, goal, timeout_s, manipulation=False,
                clock=time.monotonic, sleep=time.sleep):
    """Send `goal`, wait for the result for at most `timeout_s` in total, cancel
    the goal on timeout. Returns an Outcome; never raises for action failures.
    `manipulation=True` also requires result.success."""
    deadline = clock() + timeout_s
    if not client.wait_for_server(timeout_sec=min(SERVER_WAIT_S, timeout_s)):
        return Outcome(False, 'action server not available')

    send_future = client.send_goal_async(goal)
    if not _wait(send_future, deadline, clock, sleep):
        send_future.add_done_callback(_cancel_when_accepted)
        return Outcome(False, f'timed out after {timeout_s:g} s')
    handle = send_future.result()
    if not handle.accepted:
        return Outcome(False, 'goal rejected')

    result_future = handle.get_result_async()
    if not _wait(result_future, deadline, clock, sleep):
        handle.cancel_goal_async()
        return Outcome(False, f'timed out after {timeout_s:g} s')
    wrapped = result_future.result()
    if wrapped.status != STATUS_SUCCEEDED:
        name = _STATUS_NAMES.get(wrapped.status, f'status {wrapped.status}')
        return Outcome(False, f'action {name}')
    if manipulation and not wrapped.result.success:
        return Outcome(False, f'manipulation reported failure: {wrapped.result.message}')
    return Outcome(True)
