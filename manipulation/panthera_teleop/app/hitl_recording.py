"""Active-control recording on a policy-paced clock, with host times preserved."""
from recording import EpisodeStore
from bisect import bisect_left
from collections import deque


def data_timestamp(row):
    return row.get('data_monotonic_ns', row['monotonic_ns'])


def data_origin(metadata):
    return metadata.get('start_data_monotonic_ns', metadata['start_monotonic_ns'])


class RolloutStore(EpisodeStore):
    def submit(self, kind, payload):
        # Cameras stay live for inference and preview. Selected snapshots travel
        # with measured states so asynchronous camera events cannot leak holds.
        if kind != 'frame':
            super().submit(kind, payload)

    def collect(self, payload):
        source = payload['control_source']
        if source not in ('human', 'policy', 'hold'):
            raise ValueError('Invalid control ownership')
        transitions = self.active['metadata'].setdefault('control_transitions', [])
        changed = not transitions or transitions[-1]['source'] != source
        if changed:
            transitions.append(dict(source=source, monotonic_ns=payload['monotonic_ns'],
                                    sequence=payload['sequence']))
            self.active['metadata']['human_interventions'] = sum(t['source'] == 'human' for t in transitions)
        if source == 'hold':
            self.previous = None
            self.update(collecting=False, pause_reason='both_arms_held', data_clock_rate=0.)
            return
        rate = payload.get('policy_clock_rate', 1.) if source == 'policy' else 1.
        if not 0 < rate <= 1.000001:
            raise ValueError('Invalid policy recording clock rate')
        previous = self.previous
        new_segment = previous is None or previous['control_source'] != source
        if new_segment:
            self.segment += 1
            self.clock_history.clear()
            # A small gap keeps timestamps strictly increasing across omitted
            # holds; segment IDs prevent correction actions crossing that gap.
            if self.last_written is not None:
                self.clock_ns += round(1e9 / self.config['control_hz'])
        elif source == 'policy':
            if previous['inference_sequence'] == payload['inference_sequence']:
                delta = payload['policy_time_s'] - previous['policy_time_s']
            else:
                delta = payload['policy_time_s']
            if delta < 0:
                raise ValueError('Policy recording clock moved backwards')
            self.clock_ns += round(delta * 1e9)
        else:
            self.clock_ns += payload['monotonic_ns'] - previous['monotonic_ns']
        self.previous = payload
        stamp = self.active['start_ns'] + self.clock_ns
        self.clock_history.append((payload['monotonic_ns'], stamp))
        while len(self.clock_history) > 2 and self.clock_history[1][0] < payload['monotonic_ns'] - 1_000_000_000:
            self.clock_history.popleft()
        self.update(collecting=True, pause_reason=None, data_clock_rate=rate)
        period = 1e9 / self.config['control_hz']
        if self.last_written is not None and stamp - self.last_written < period - 1:
            return
        sample = {k: v for k, v in payload.items() if k != 'camera_frames'}
        sample.update(data_monotonic_ns=stamp, data_time_s=self.clock_ns / 1e9,
                      recording_segment=self.segment, data_clock_rate=rate)
        self.write('sample', sample)
        self.last_written = stamp
        # Duration shown in the dashboard is useful training time, not wait time.
        self.update(duration=self.clock_ns / 1e9)
        for name, frame in payload.get('camera_frames', {}).items():
            last_stamp, last_sequence = self.last_frames.get(name, (None, None))
            if frame['sequence'] == last_sequence:
                continue
            history = list(self.clock_history)
            received = frame['monotonic_ns']
            if not history[0][0] <= received <= history[-1][0]:
                continue
            right = bisect_left([t[0] for t in history], received)
            if right == 0:
                frame_stamp = history[0][1]
            else:
                a, b = history[right-1], history[right]
                frame_stamp = a[1] + round((b[1]-a[1]) * (received-a[0]) / (b[0]-a[0]))
            if last_stamp is not None and frame_stamp - last_stamp < 1e9 / self.config['camera_fps'] - 1:
                continue
            self.write('frame', dict(frame, data_monotonic_ns=frame_stamp,
                                    data_time_s=(frame_stamp-self.active['start_ns']) / 1e9,
                                    recording_segment=self.segment))
            self.last_frames[name] = (frame_stamp, frame['sequence'])

    def handle(self, kind, payload):
        if kind == 'rollout_begin':
            if self.active:
                raise RuntimeError('A rollout is already recording')
            self.pre_roll.clear()
            self.context = dict(self.context, rollout=payload['details'])
            self.start(payload['sample'])
            self.clock_ns = 0
            self.previous = self.last_written = None
            self.segment = -1
            self.clock_history = deque()
            self.last_frames = {}
            self.active['metadata'].update(schema_version=2, start_data_monotonic_ns=self.active['start_ns'],
                data_timing='Active control only; policy trajectory time during policy execution, host time during human guidance',
                timing='Original host receive timestamps in monotonic_ns; training clock in data_monotonic_ns')
            self.update(collecting=False, pause_reason='both_arms_held', data_clock_rate=0.)
            return
        if kind == 'rollout_end':
            if self.active:
                self.active['metadata']['outcome'] = payload
                self.finish(payload['label'], payload.get('reason', 'operator'),
                            complete=payload.get('complete', True))
            return
        if kind == 'handoff_event':
            if self.active:
                self.active['metadata'].setdefault('handoff_events', []).append(payload)
            return
        if kind == 'close':
            return super().handle(kind, payload)
        # No motion trigger or pre-roll: reset/idle movement is never a rollout.
        if self.active and kind in ('sample', 'frame'):
            if payload['monotonic_ns'] < self.active['start_ns']:
                return
            if kind == 'sample':
                self.collect(payload)

    def finish(self, *args, **kwargs):
        if self.active:
            self.active['metadata']['data_duration_s'] = self.clock_ns / 1e9
            self.active['metadata']['end_data_monotonic_ns'] = self.active['start_ns'] + self.clock_ns
        super().finish(*args, **kwargs)
        self.update(collecting=False, pause_reason=None, data_clock_rate=0.)
