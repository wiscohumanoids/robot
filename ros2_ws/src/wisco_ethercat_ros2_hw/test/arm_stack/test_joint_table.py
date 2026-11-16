"""arm01's joint table against the actuator datasheets (config/actuator_models.yaml)."""

import os

import pytest
import yaml

from arm_stack.config import load_arm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARM01_DIR = os.path.normpath(os.path.join(ROOT, '..', '..', '..', 'robots', 'arm01'))
ARM01 = os.path.join(ARM01_DIR, 'config', 'arm01.yaml')
MODELS = os.path.join(ROOT, 'config', 'actuator_models.yaml')


def models():
    with open(MODELS) as f:
        return yaml.safe_load(f)['models']


def test_arm01_has_no_nulls_and_configured_clamps():
    arm = load_arm(ARM01)
    table = models()
    for j in arm.joints:
        m = table[j.model]
        assert j.nulls == [], (j.name, j.nulls)
        assert j.max_torque_nm == pytest.approx(m['peak_effort_nm'])
        assert j.max_vel_rad_s == pytest.approx(m['bringup_velocity_limit_rad_s'])
        assert j.rated_current_a == pytest.approx(m['rated_current_a'])
        assert j.counts_per_rev == m['counts_per_rev']


@pytest.mark.parametrize('name', ['x4_10', 'x4_36', 'x6_60'])
def test_model_scales_follow_kt_and_rated_current(name):
    m = models()[name]
    # 0x6071 / 0x6077 are per mille of rated current
    assert m['effort_nm_per_permille'] == pytest.approx(
        m['kt_output_nm_per_a'] * m['rated_current_a'] / 1000, rel=2e-3)
    # the drive-side 0x6072 cap sits just above the controller's bring-up clamp
    cap = m['max_torque_permille'] * m['effort_nm_per_permille']
    assert m['bringup_effort_limit_nm'] < cap < m['peak_effort_nm']
    assert m['max_velocity_rad_s'] < m['no_load_velocity_rad_s']


def test_clamp_above_datasheet_is_refused(tmp_path):
    with open(ARM01) as f:
        raw = yaml.safe_load(f)
    raw['urdf'] = os.path.join(ARM01_DIR, 'urdf', 'arm01.urdf')
    raw['joints'][0]['max_vel_rad_s'] = 30.0     # x6_60 no-load speed is 18.4 rad/s
    path = tmp_path / 'arm.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='max_vel_rad_s 30 is above'):
        load_arm(str(path))
