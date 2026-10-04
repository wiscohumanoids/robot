"""The "succeeded within N s?" decision: call_action() with fake futures and a fake
clock (no ROS, no real waiting). Everything the ROS wrapper does with an action
client is covered here: server missing, goal rejected, timeouts (and cancelling),
non-success results, manipulation failures, and a goal accepted after we gave up."""
from types import SimpleNamespace

import pytest

from behavior_tree.action_call import (POLL_S, SERVER_WAIT_S, STATUS_ABORTED, STATUS_CANCELED,
                                        STATUS_SUCCEEDED, call_action)

NEVER = float('inf')


class Clock:
    """Fake time: sleeping advances it, so a 60 s timeout takes no real time."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Future:
    def __init__(self, clock, value, ready_at=0.0):
        self.clock, self.value, self.ready_at = clock, value, ready_at
        self.callbacks = []

    def done(self):
        return self.clock.t >= self.ready_at

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def finish(self):
        """Simulate the future completing later (rclpy would invoke the callbacks)."""
        self.ready_at = 0.0
        for callback in self.callbacks:
            callback(self)


class Handle:
    def __init__(self, clock, accepted=True, result=None, result_at=0.0):
        self.accepted = accepted
        self.result_future = Future(clock, result, result_at)
        self.cancels = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancels += 1


class Client:
    def __init__(self, clock, handle=None, server=True, send_at=0.0):
        self.server = server
        self.server_waits = []
        self.goals = []
        self.send_future = Future(clock, handle, send_at)

    def wait_for_server(self, timeout_sec):
        self.server_waits.append(timeout_sec)
        return self.server

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.send_future


def wrapped(status=STATUS_SUCCEEDED, success=True, message=''):
    return SimpleNamespace(status=status, result=SimpleNamespace(success=success, message=message))


def make(handle_kwargs=None, send_at=0.0, server=True):
    clock = Clock()
    handle = Handle(clock, **(handle_kwargs or {}))
    client = Client(clock, handle, server, send_at)
    return clock, handle, client


def call(client, clock, timeout_s=60.0, manipulation=False):
    return call_action(client, 'GOAL', timeout_s, manipulation, clock=clock, sleep=clock.sleep)


def test_status_constants_match_action_msgs_goal_status():
    assert (STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED) == (4, 5, 6)


# ---------------------------------------------------------------- the happy path
def test_success_sends_the_goal_and_returns_ok():
    clock, handle, client = make({'result': wrapped()})
    out = call(client, clock)
    assert out.ok and out.reason == '' and out.retryable
    assert client.goals == ['GOAL'] and handle.cancels == 0


def test_a_slow_but_in_time_result_is_a_success():
    clock, handle, client = make({'result': wrapped(), 'result_at': 59.0})
    assert call(client, clock).ok and handle.cancels == 0


def test_a_slow_goal_acceptance_that_is_still_in_time_is_a_success():
    clock, handle, client = make({'result': wrapped()}, send_at=30.0)
    assert call(client, clock).ok


# ---------------------------------------------------------------- server missing / goal rejected
def test_no_server_fails_without_sending_anything():
    clock, handle, client = make(server=False)
    out = call(client, clock)
    assert (out.ok, out.reason) == (False, 'action server not available')
    assert client.goals == []


@pytest.mark.parametrize('timeout_s, expected_wait', [(60.0, SERVER_WAIT_S), (2.0, 2.0), (0.5, 0.5)])
def test_server_wait_is_capped_by_the_skill_timeout(timeout_s, expected_wait):
    clock, handle, client = make({'result': wrapped()})
    call(client, clock, timeout_s)
    assert client.server_waits == [expected_wait]


def test_rejected_goal_fails_and_nothing_is_cancelled():
    clock, handle, client = make({'accepted': False})
    out = call(client, clock)
    assert (out.ok, out.reason) == (False, 'goal rejected')
    assert handle.cancels == 0


# ---------------------------------------------------------------- timeouts
def test_result_that_never_arrives_times_out_and_cancels_the_goal_once():
    clock, handle, client = make({'result': wrapped(), 'result_at': NEVER})
    out = call(client, clock)
    assert (out.ok, out.reason) == (False, 'timed out after 60 s')
    assert out.retryable
    assert handle.cancels == 1
    assert 60.0 < clock.t < 60.0 + 2 * POLL_S          # gave up right after the limit, not before


def test_result_arriving_after_the_limit_is_a_timeout():
    clock, handle, client = make({'result': wrapped(), 'result_at': 61.0})
    out = call(client, clock)
    assert out.reason == 'timed out after 60 s' and handle.cancels == 1


def test_the_time_limit_covers_sending_and_waiting_together():
    clock, handle, client = make({'result': wrapped(), 'result_at': 80.0}, send_at=50.0)
    out = call(client, clock)                            # 50 s to accept + result at 80 s > 60 s
    assert out.reason == 'timed out after 60 s' and handle.cancels == 1


def test_sending_and_waiting_that_fit_together_succeed():
    clock, handle, client = make({'result': wrapped(), 'result_at': 55.0}, send_at=50.0)
    assert call(client, clock).ok


@pytest.mark.parametrize('timeout_s, text', [(60.0, '60 s'), (60, '60 s'), (2.5, '2.5 s'), (0.5, '0.5 s')])
def test_timeout_message_shows_the_limit(timeout_s, text):
    clock, handle, client = make({'result': wrapped(), 'result_at': NEVER})
    assert call(client, clock, timeout_s).reason == f'timed out after {text}'


def test_a_shorter_limit_times_out_sooner():
    clock, handle, client = make({'result': wrapped(), 'result_at': 10.0})
    assert call(client, clock, timeout_s=5.0).reason == 'timed out after 5 s'


def test_goal_acceptance_that_never_arrives_times_out():
    clock, handle, client = make({'result': wrapped()}, send_at=NEVER)
    out = call(client, clock)
    assert (out.ok, out.reason) == (False, 'timed out after 60 s')
    assert handle.cancels == 0                           # nothing to cancel yet: no handle


def test_a_goal_accepted_after_we_gave_up_is_cancelled():
    clock, handle, client = make({'result': wrapped()}, send_at=NEVER)
    call(client, clock)
    client.send_future.finish()                          # the server finally accepts it
    assert handle.cancels == 1


def test_a_goal_rejected_after_we_gave_up_is_not_cancelled():
    clock, handle, client = make({'accepted': False}, send_at=NEVER)
    call(client, clock)
    client.send_future.finish()
    assert handle.cancels == 0


def test_a_late_send_that_produced_no_handle_is_ignored():
    clock = Clock()
    client = Client(clock, handle=None, send_at=NEVER)
    call(client, clock)
    client.send_future.finish()                          # must not raise


# ---------------------------------------------------------------- results that are not success
@pytest.mark.parametrize('status, text', [
    (STATUS_ABORTED, 'action aborted'),
    (STATUS_CANCELED, 'action canceled'),
    (2, 'action status 2'),      # still executing
    (0, 'action status 0'),      # unknown
    (7, 'action status 7'),
])
def test_non_success_statuses_become_failures_with_a_readable_reason(status, text):
    clock, handle, client = make({'result': wrapped(status=status)})
    out = call(client, clock)
    assert (out.ok, out.reason) == (False, text)
    assert out.retryable and handle.cancels == 0


def test_an_aborted_manipulation_reports_the_status_not_the_success_flag():
    clock, handle, client = make({'result': wrapped(status=STATUS_ABORTED, success=True)})
    assert call(client, clock, manipulation=True).reason == 'action aborted'


# ---------------------------------------------------------------- manipulation result
def test_manipulation_success_flag_true_is_ok():
    clock, handle, client = make({'result': wrapped(success=True, message='done')})
    assert call(client, clock, manipulation=True).ok


def test_manipulation_success_flag_false_fails_with_its_message():
    clock, handle, client = make({'result': wrapped(success=False, message='grasp failed')})
    out = call(client, clock, manipulation=True)
    assert (out.ok, out.reason) == (False, 'manipulation reported failure: grasp failed')


def test_manipulation_failure_with_an_empty_message():
    clock, handle, client = make({'result': wrapped(success=False, message='')})
    assert call(client, clock, manipulation=True).reason == 'manipulation reported failure: '


def test_navigation_ignores_the_success_flag_it_does_not_have():
    clock, handle, client = make({'result': SimpleNamespace(status=STATUS_SUCCEEDED, result=SimpleNamespace())})
    assert call(client, clock, manipulation=False).ok       # NavigateToPose results have no `success`


def test_navigation_ignores_a_false_success_flag():
    clock, handle, client = make({'result': wrapped(success=False)})
    assert call(client, clock, manipulation=False).ok


def test_manipulation_requires_the_flag_to_exist():
    clock, handle, client = make({'result': SimpleNamespace(status=STATUS_SUCCEEDED, result=SimpleNamespace())})
    with pytest.raises(AttributeError):                    # a malformed result is a bug, run_skill turns it into a failure
        call(client, clock, manipulation=True)
