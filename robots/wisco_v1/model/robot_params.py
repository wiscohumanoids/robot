"""Reads the Wisco V1 numbers and derives every per-joint value the sims need.

Plain Python + numpy + pyyaml (no mjlab import), so the Mac checks (robots/wisco_v1/checks/) and the mjlab task
(training/locomotion/wisco_v1/) both read the same numbers.

Where the numbers come from (change them there, then rerun sync_robot_folder.py, build_training_model.py, checks/run_checks.py):
  robot_numbers.yaml   THE numbers file: motors (datasheet values + sources), delay/bus, mass overrides,
                       control (gains, modes, default pose, locomotion action joints), sensors (IMU, camera)
  ankle_linkage.json   ankle linkage Jacobians (linkage_tools.py, from robots/wisco_v1/linkage.json)
  robots/wisco_v1/linkage.py     waist linkage Jacobian
  robots/wisco_v1/model_report.json  joint ranges (from CAD / ROM sheet)

Joint values from motor values:
  single motor:  joint = motor
  linkage (2 motors through rods, J = d(motor angle)/d(joint angle) at the zero pose):
     armature, viscous damping: sum_k J_kj^2 * motor value
     Coulomb friction, torque:  sum_k |J_kj| * motor value
     speed limit:               motor no-load speed / max_k |J_kj|

Torque-speed curve (back-EMF limit, per motor, output side):
     max torque(w) = min(peak, slope * (w_noload - |w|)),  slope = Kt * Ke * gear / R
     (slope from datasheet motor constants; X8 cross-checked against the X8-120 dyno curve: 42.5 vs 38.1)

PD gains: RP1-style values from robot_numbers.yaml (control.gains); action scale = 0.25 * peak / kp.
"""
import functools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBOT_DIR = HERE.parent                              # robots/wisco_v1 (the robot model, read-only)
SOURCE_XML = ROBOT_DIR / "robot.xml"
REPORT = ROBOT_DIR / "model_report.json"   # joint ranges are the CAD export's; only ankle pitch lower is changed (ankle_pitch_lower())
TRAIN_XML = HERE / "wisco_v1_train.xml"               # written by build_training_model.py
LINKAGE_OUT = HERE / "ankle_linkage.json"             # written by linkage_tools.py
NUMBERS = HERE / "robot_numbers.yaml"

# Held still in training: removed from the training model (welded), as agreed.
FIXED_JOINTS = ("head_yaw", "head_pitch")

# Ankle pitch lower limit from the range-of-motion sheet; the linkage analysis scans up from here
SOURCE_ANKLE_PITCH_LOWER = -0.610865238


@functools.lru_cache(maxsize=1)
def load_numbers():
    """The numbers file."""
    import yaml
    return yaml.safe_load(NUMBERS.read_text())


# Default (home) pose: knees bent like mjlab's G1 KNEES_BENT pose.
DEFAULT_POSE = {k: v for k, v in load_numbers()["control"]["default_pose"].items() if k != "source"}
# Joints the walking policy drives (legs + waist; arms belong to the ROS controls stack).
LOCOMOTION_ACTION_JOINTS = tuple(load_numbers()["control"]["locomotion_action_joints"])

TORSO_BODY = "torso"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

RPM = 2 * math.pi / 60


def load_report():
    """Per-joint values from model_report.json."""
    return json.loads(REPORT.read_text())["joints"]


def load_linkage_info():
    """The ankle linkage analysis, if it has been run."""
    return json.loads(LINKAGE_OUT.read_text()) if LINKAGE_OUT.exists() else None


def load_motor_specs():
    """Motor section of robot_numbers.yaml, in the layout the old motor_specs.json had."""
    n = load_numbers()
    return {"models": n["motors"], "joint_to_motor": n["joint_to_motor"], "delay": n["delay"]}


def load_mass_overrides():
    """{link name: {"mass_kg": value, "com": [...], "source": "..."}} from robot_numbers.yaml masses.overrides."""
    return dict(load_numbers()["masses"].get("overrides") or {})


def gains(joint):
    """(kp, kd) for a joint name (left_/right_ share one entry), from robot_numbers.yaml control.gains."""
    key = joint.replace("left_", "").replace("right_", "")
    g = load_numbers()["control"]["gains"][key]
    return float(g["kp"]), float(g["kd"])


def sensors():
    """Sensor poses from the numbers file."""
    return load_numbers()["sensors"]


def ankle_pitch_lower():
    """Training lower limit of ankle pitch (from the linkage analysis when available)."""
    info = load_linkage_info()
    return info["training_ankle_pitch_lower"] if info else SOURCE_ANKLE_PITCH_LOWER


def motor(name):
    """Output-side motor values used by the sim, from robot_numbers.yaml (motors)."""
    m = load_motor_specs()["models"][name]
    v = lambda k: m[k]["value"]
    N = v("gear_ratio")
    w_nl = v("no_load_speed_rpm") * RPM
    return {
        "model": name,
        "armature": v("rotor_inertia_kgcm2") * 1e-4 * N ** 2,      # reflected rotor inertia, kg m^2
        "armature_range": [x * 1e-4 * N ** 2 for x in m["rotor_inertia_kgcm2"].get("range", [v("rotor_inertia_kgcm2")] * 2)],
        "peak": v("peak_torque_nm"),
        "rated": v("rated_torque_nm"),
        "w_noload": w_nl,
        "slope": None,                                             # filled below
        "friction": v("backdrive_torque_nm"),
        "damping": v("viscous_damping_nms"),
        "damping_upper": m["viscous_damping_nms"]["upper_bound"],
        "mass": v("mass_kg"),
    }


def _torque_speed_slope(name):
    """N*m per output rad/s: Kt * Ke * gear / R (datasheet motor constants)."""
    params = {"X4-10": (6, 0.32), "X4-36": (6, 0.39), "X6-60": (16, 0.41), "X8-150": (13.74, 0.18)}  # Ke Vdc/krpm, R line ohm (datasheets)
    m = load_motor_specs()["models"][name]
    Ke, R = params[name]
    ke_out = Ke / 1000 / RPM * m["gear_ratio"]["value"]
    return m["torque_constant_nm_per_a"]["value"] * ke_out / R


def linkage_jacobians():
    """{joint pair: (J 2x2 rows=motors cols=joints, motor model)} at the zero pose."""
    out = {}
    info = load_linkage_info()
    for side in ("left", "right"):
        J = info["sides"][side]["jacobian_at_zero_pose"]
        out[(f"{side}_ankle_pitch", f"{side}_ankle_roll")] = (J, "X4-36")
    sys.path.insert(0, str(ROBOT_DIR))
    sys.dont_write_bytecode = True
    from linkage import Mechanism                  # robots/wisco_v1/linkage.py converges fine at the zero pose
    Jw = Mechanism("waist").jacobian((0.0, 0.0)).tolist()
    out[("torso_roll", "torso_pitch")] = (Jw, "X6-60")
    return out


def joint_params():
    """Per policy joint: everything the sim needs, derived from robot_numbers.yaml."""
    rep = load_report()
    specs = load_motor_specs()["joint_to_motor"]
    single = {j: m for m, joints in specs.items() if not m.startswith(("_", "linkage")) for j in joints}
    lj = linkage_jacobians()
    out = {}

    def finish(name, d):
        d["kp"], d["kd"] = gains(name)
        d["action_scale"] = 0.25 * d["effort"] / d["kp"]
        lower, upper = rep[name]["lower"], rep[name]["upper"]
        if name.endswith("ankle_pitch"):
            lower = ankle_pitch_lower()
        d.update(lower=lower, upper=upper, default=DEFAULT_POSE.get(name, 0.0))
        out[name] = d

    for name, mname in single.items():
        if name in FIXED_JOINTS:
            continue
        mo = motor(mname)
        slope = _torque_speed_slope(mname)
        finish(name, {
            "motor": mname, "armature": mo["armature"], "armature_range": mo["armature_range"],
            "effort": mo["peak"], "rated": mo["rated"], "velocity": mo["w_noload"],
            "frictionloss": mo["friction"], "damping": mo["damping"], "damping_upper": mo["damping_upper"],
            # mjlab DC motor model: max torque = saturation * (1 - |w| / velocity_limit), capped at effort
            "dc_saturation": slope * mo["w_noload"], "dc_velocity_limit": mo["w_noload"],
        })
    for (ja, jb), (J, mname) in lj.items():
        mo = motor(mname)
        slope = _torque_speed_slope(mname)
        for c, name in enumerate((ja, jb)):
            col = [abs(J[0][c]), abs(J[1][c])]
            sq = col[0] ** 2 + col[1] ** 2
            sat = sum(col) * slope * mo["w_noload"]
            finish(name, {
                "motor": f"2x{mname}", "armature": sq * mo["armature"],
                "armature_range": [sq * x for x in mo["armature_range"]],
                "effort": sum(col) * mo["peak"], "rated": sum(col) * mo["rated"],
                "velocity": mo["w_noload"] / max(col),
                "frictionloss": sum(col) * mo["friction"], "damping": sq * mo["damping"],
                "damping_upper": sq * mo["damping_upper"],
                "dc_saturation": sat, "dc_velocity_limit": sat / (sq * slope),
            })
    return out


# Per-motor limits used by the ankle linkage penalty (X4-36 datasheet).
def _x4_36():
    """X4-36 peak and rated torque."""
    m = load_motor_specs()["models"]["X4-36"]
    return m["peak_torque_nm"]["value"], m["rated_torque_nm"]["value"]


X4_36_PEAK_NM, X4_36_RATED_NM = _x4_36()


def command_delay_ms():
    """Command delay range, ms."""
    return load_motor_specs()["delay"]["command_delay_ms"]["range"]


if __name__ == "__main__":
    for n, p in joint_params().items():
        print(f"{n:22s} {p['motor']:9s} arm {p['armature']:.4f}  kp {p['kp']:6.1f}  kd {p['kd']:5.2f}  "
              f"peak {p['effort']:6.1f}  v_nl {p['velocity']:5.2f}  fric {p['frictionloss']:.2f}  damp {p['damping']:.3f}  "
              f"tau(w)={p['dc_saturation']:.0f}(1-w/{p['dc_velocity_limit']:.1f})  scale {p['action_scale']:.3f}")
