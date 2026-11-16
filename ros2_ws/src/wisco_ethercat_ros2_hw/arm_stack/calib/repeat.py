"""`arm repeat [--indicators]`: ISO 9283 pose repeatability, measured only.

Warm-up (20 min of motion on hardware, 2 s in sim), then 5 poses x 30 cycles
from a fixed home (quick: 6 cycles) with the tuned settings from the joint
config (`repeatability_tuning`, written by mocap-repeat) or the defaults.
RP = mean(l) + 3 S_l per pose; drift = the linear trend across cycles.

Source: --indicators (or no mocap on the client) uses the dial-indicator cube
(hw/indicator_cube.py): one pose per fixture setup, translation only; on
hardware the stand is set and zeroed at each pose and the three readings are
typed per cycle (or read from digital indicators). Otherwise the mocap.
`arm mocap-repeat` is the version that also separates the effects and tunes
the settings, which needs the mocap.

Pass: RP under 0.5 mm per pose, drift under 0.25 mm, source 3 sigma under 0.1 mm.
Writes validation.repeatability_rp_mm.
"""

from __future__ import annotations

import numpy as np

from ..hw import indicator_cube
from . import mocap_repeat as rep
from .common import StageContext, StageResult


def run(ctx: StageContext, indicators: bool = False, cycles: int = None, warm_s: float = None) -> StageResult:
    mocap = None if indicators else ctx.client.mocap()
    source = mocap if mocap is not None else indicator_cube.for_context(ctx)
    cycles = cycles or (6 if ctx.quick else 30)
    lines, flags = [], []
    if mocap is None and not indicators:
        lines.append('no mocap on this client: using the dial-indicator cube')
    if isinstance(source, indicator_cube.IndicatorCube):
        c = source.cfg
        lines.append('cube %.0f mm on the flange, indicator axes %s in base_link, resolution %.1f um'
                     % (c.cube_size_m * 1e3, np.round(c.axes, 3).tolist(), c.resolution_m * 1e6))
    res = rep.iso_measure(ctx, source, cycles, np.random.default_rng(11), warm_s)
    rp, dr = res['rp_mm'], res['drift_mm']
    lines.append('%s: 3 sigma %.4f mm, %d poses x %d cycles' % (res['source'], res['noise_3sigma_m'] * 1e3,
                                                                len(rp), cycles))
    lines.append('RP per pose %s mm (pass < %.1f), drift %s mm (pass < %.2f)' % (
        np.round(rp, 4).tolist(), rep.RP_MM, np.round(dr, 4).tolist(), rep.DRIFT_MM))
    if res['noise_3sigma_m'] >= 1e-4:
        flags.append('source 3 sigma %.3f mm is over 0.1 mm' % (res['noise_3sigma_m'] * 1e3))
    metrics = {'rp_mm': rp.tolist(), 'drift_mm': dr.tolist(), 'cycles': cycles, 'source': res['source'],
               'noise_3sigma_mm': res['noise_3sigma_m'] * 1e3,
               'time_since_power_on_s': ctx.client.time_since_power_on()}
    return StageResult('repeat', res['passed'], metrics,
                       {'validation': {'repeatability_rp_mm': float(np.max(rp))}}, lines, flags)
