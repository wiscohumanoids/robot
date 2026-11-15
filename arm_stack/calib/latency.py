"""Stage 7, `arm latency`: command-to-motion delay.

Per joint:

  * a chirp in Mode P (the spec's measure): the cross-correlation lag between
    commanded and measured position. It includes the drive's own servo lag,
    so it is reported with a fitted second-order servo model, not used alone;
  * the transport delay on the torque channel: a torque chirp in Mode T and
    the cross-correlation lag between the torque the RT loop sent (0x6071)
    and the torque the drive reports (0x6077). The current loop is far faster
    than the bus, so this lag is the bus + drive transport delay without the
    position servo in the way. latency_ms is this number (median over
    joints); PACE models the servo explicitly and compares its fitted delay
    with it.

Time to first motion is not used: it measures stiction, not delay.
Above 3 ms is flagged.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import Trace, chirp, chirp_profile, move_to
from ..hw.client import Command
from .common import DEG, StageContext, StageResult, safe_home

FLAG_MS = 3.0


def xcorr_lag(cmd, meas, dt, max_lag_s=0.05):
    a = cmd - cmd.mean()
    b = meas - meas.mean()
    max_lag = int(max_lag_s / dt)
    lags = np.arange(0, max_lag + 1)
    c = [np.dot(a[:len(a) - k], b[k:]) for k in lags]
    k = int(np.argmax(c))
    # parabolic refinement
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    return k * dt


def frequency_response(cmd, meas, dt, f_lo, f_hi):
    n = len(cmd)
    win = np.hanning(n)
    U = np.fft.rfft((cmd - cmd.mean()) * win)
    Y = np.fft.rfft((meas - meas.mean()) * win)
    f = np.fft.rfftfreq(n, dt)
    band = (f >= f_lo) & (f <= f_hi) & (np.abs(U) > 0.05 * np.abs(U).max())
    return f[band], Y[band] / U[band]


def fit_delay(f, H, dt):
    """Fit H(jw) = wn^2 / (s^2 + 2 z wn s + wn^2) * exp(-s T); returns T, wn, z."""
    w = 2 * np.pi * f
    s = 1j * w
    best = (np.inf, 0.0, 0.0, 0.0)
    for wn in np.geomspace(2 * np.pi * 2, 2 * np.pi * 200, 60):
        for z in (0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0):
            G = wn ** 2 / (s ** 2 + 2 * z * wn * s + wn ** 2)
            # remaining phase is -w T; solve T in least squares on unwrapped phase
            ph = np.unwrap(np.angle(H / G))
            T = max(0.0, -np.dot(w, ph) / np.dot(w, w))
            model = G * np.exp(-s * T)
            err = np.sum(np.abs(np.log(H / model)) ** 2)
            if err < best[0]:
                best = (err, T, wn, z)
    return best[1], best[2], best[3]


def torque_channel_delay(ctx, j, center_q, duration=2.0, f0=1.0, f1=40.0):
    """Lag between the torque sent and the torque measured, Mode T, seconds."""
    client, arm = ctx.client, ctx.arm
    n = arm.n
    move_to(client, center_q, max_speed=0.4, settle=0.3)
    kp = np.array([max(s.kp, 5.0) for s in arm.joints])
    kd = np.array([max(s.kd, 0.2) for s in arm.joints])
    amp = 0.08 * arm.joints[j].max_torque_nm
    t, x, _ = chirp_profile(amp, f0, f1, duration, client.cmd_dt)
    sent, meas = [], []
    for k in range(len(t)):
        ff = np.zeros(n)
        ff[j] = x[k]
        client.send(Command(mode='T', q=center_q, qd=np.zeros(n), kp=kp, kd=kd, tau_ff=ff))
        s = client.step(client.cmd_dt)
        if s.tau_cmd is None:
            return None
        sent.append(s.tau_cmd[j])
        meas.append(s.tau[j])
    client.hold(0.3)
    return xcorr_lag(np.array(sent), np.array(meas), client.cmd_dt, max_lag_s=0.02)


def run(ctx: StageContext) -> StageResult:
    client, model, arm = ctx.client, ctx.model, ctx.arm
    home = safe_home(model)
    dt = client.cmd_dt
    duration = 4.0 if ctx.quick else 10.0
    lines, flags, metrics = [], [], {}
    transport, xc = [], []
    for j, spec in enumerate(arm.joints):
        span = model.upper[j] - model.lower[j]
        amp = min(3 * DEG, 0.1 * span)
        center = float(np.clip(home[j], model.lower[j] + amp + 3 * DEG, model.upper[j] - amp - 3 * DEG))
        f1 = 8.0
        tr = Trace()
        chirp(client, j, center, amp, 0.2, f1, duration, trace=tr, max_vel=0.7 * spec.max_vel_rad_s)
        a = tr.arrays()
        cmd, meas = a['q_cmd'][:, j], a['q'][:, j]
        lag = xcorr_lag(cmd, meas, dt)
        f, H = frequency_response(cmd, meas, dt, 0.3, f1)
        T_fit, wn, z = fit_delay(f, H, dt)
        q_c = home.copy()
        q_c[j] = center
        T = torque_channel_delay(ctx, j, q_c)
        if T is None:
            flags.append('client reports no sent torque; transport delay from the servo fit instead')
            T = T_fit
        transport.append(T)
        xc.append(lag)
        metrics[spec.name] = {'xcorr_lag_ms': lag * 1e3, 'transport_ms': T * 1e3,
                              'servo_fit_delay_ms': T_fit * 1e3,
                              'servo_wn_hz': wn / (2 * np.pi), 'servo_zeta': z}
        lines.append('%-15s transport %.2f ms (torque channel), position lag %.2f ms, servo %.1f Hz '
                     'zeta %.2f' % (spec.name, T * 1e3, lag * 1e3, wn / (2 * np.pi), z))
        move_to(client, home, max_speed=0.4)
    latency_ms = float(np.median(transport) * 1e3)
    spread = float((np.max(transport) - np.min(transport)) * 1e3)
    lines.append('latency %.2f ms (median over joints, spread %.2f ms); xcorr median %.2f ms'
                 % (latency_ms, spread, np.median(xc) * 1e3))
    if latency_ms > FLAG_MS:
        flags.append('latency %.2f ms is above %.0f ms; check the cycle time, DC sync and host jitter'
                     % (latency_ms, FLAG_MS))
    metrics['latency_ms'] = latency_ms
    metrics['xcorr_ms'] = float(np.median(xc) * 1e3)
    return StageResult('latency', True, metrics, {'latency_ms': latency_ms}, lines, flags)
