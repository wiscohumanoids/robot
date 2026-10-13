"""The "succeeded within N s?" decision: call_action() with fake futures and a fake
clock (no ROS, no real waiting). Everything the ROS wrapper does with an action
client is covered here: server missing, goal rejected, timeouts (and cancelling),
non-success results, manipulation failures, and a goal accepted after we gave up."""
from types import SimpleNamespace

import pytest

from behavior_tree.action_call import (CANCEL_WAIT_S, POLL_S, SERVER_WAIT_S, STATUS_ABORTED,
                                        STATUS_CANCELED, STATUS_SUCCEEDED, STOPPED, call_action)

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
    def __init__(self, clock, accepted=True, result=None, result_at=0.0, cancel_takes=None,
                 cancel_error=None):
        self.accepted = accepted
        self.result_future = Future(clock, result, result_at)
        self.cancels = 0
        self.cancel_takes = cancel_takes
        self.cancel_error = cancel_error

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancels += 1
        if self.cancel_error is not None:
            raise self.cancel_error
        if self.cancel_takes is not None:          # the server finishes the goal this much later
            self.result_future.ready_at = self.result_future.clock.t + self.cancel_takes


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


def call(client, clock, timeout_s=60.0, manipulation=False, **kwargs):
    return call_action(client, 'GOAL', timeout_s, manipulation, clock=clock, sleep=clock.sleep, **kwargs)


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
    # cancelled right after the limit, then waited (bounded) for the server to finish the goal
    assert 60.0 + CANCEL_WAIT_S < clock.t < 60.0 + CANCEL_WAIT_S + 3 * POLL_S


def test_after_a_timeout_it_waits_until_the_server_has_finished_the_cancelled_goal():
    clock, handle, client = make({'result': wrapped(STATUS_CANCELED), 'result_at': NEVER,
                                  'cancel_takes': 0.1})   # manipulation_runner notices on its next 0.1 s tick
    out = call(client, clock)
    assert (out.ok, out.reason, out.retryable) == (False, 'timed out after 60 s', True)
    assert handle.cancels == 1 and handle.result_future.done()      # safe to send the next goal now
    assert 60.1 <= clock.t < 60.1 + 3 * POLL_S                      # and no longer than needed


def test_a_result_that_lands_during_the_cancel_wait_is_still_a_timeout():
    clock, handle, client = make({'result': wrapped(), 'result_at': 62.0})
    out = call(client, clock)
    assert (out.ok, out.reason) == (False, 'timed out after 60 s') and clock.t < 62.0 + 2 * POLL_S


def test_retrying_after_a_timeout_is_not_rejected_by_a_server_still_busy_with_the_old_goal():
    """The bug this guards: the server accepts one goal at a time and frees itself only when
    the cancelled goal has finished, so an instant retry was rejected (burning every attempt)."""
    clock = Clock()

    class OneGoalServer:
        def __init__(self):
            self.handles = []

        def wait_for_server(self, timeout_sec):
            return True

        def send_goal_async(self, goal):
            busy = any(not h.result_future.done() for h in self.handles)
            first = not self.handles
            handle = Handle(clock, accepted=not busy, result=wrapped(STATUS_CANCELED if first else STATUS_SUCCEEDED),
                            result_at=NEVER if first else 0.0, cancel_takes=0.1)
            self.handles.append(handle)
            return Future(clock, handle)

    server = OneGoalServer()
    assert call(server, clock, manipulation=True).reason == 'timed out after 60 s'
    assert call(server, clock, manipulation=True).ok               # attempt 2 is accepted


# ---------------------------------------------------------------- shutdown (stop)
def stop_at(clock, t):
    return lambda: clock.t >= t


def test_stop_cancels_the_running_goal_and_returns_at_once_without_retry():
    clock, handle, client = make({'result': wrapped(), 'result_at': NEVER})
    out = call(client, clock, stop=stop_at(clock, 10.0))
    assert (out.ok, out.reason, out.retryable) == (False, STOPPED, False)
    assert handle.cancels == 1
    assert 10.0 <= clock.t < 10.0 + 2 * POLL_S                      # no cancel wait: the node is going down


def test_stop_before_the_goal_is_accepted_cancels_it_when_it_is():
    clock, handle, client = make({'result': wrapped()}, send_at=NEVER)
    out = call(client, clock, stop=stop_at(clock, 1.0))
    assert (out.reason, out.retryable) == (STOPPED, False) and handle.cancels == 0
    client.send_future.finish()
    assert handle.cancels == 1


def test_stop_with_a_cancel_that_cannot_be_sent_says_so_instead_of_raising():
    clock, handle, client = make({'result': wrapped(), 'result_at': NEVER,
                                  'cancel_error': RuntimeError('context is not valid')})
    out = call(client, clock, stop=stop_at(clock, 1.0))
    assert not out.ok and not out.retryable
    assert out.reason == f'{STOPPED} (cancel not sent: context is not valid)'


def test_stop_during_the_cancel_wait_ends_the_wait():
    clock, handle, client = make({'result': wrapped(), 'result_at': NEVER})
    out = call(client, clock, stop=stop_at(clock, 61.0))
    assert out.reason == 'timed out after 60 s' and clock.t < 61.0 + 2 * POLL_S


def test_a_goal_that_finishes_is_unaffected_by_a_stop_flag_that_stays_false():
    clock, handle, client = make({'result': wrapped(), 'result_at': 5.0})
    assert call(client, clock, stop=lambda: False).ok and handle.cancels == 0


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
