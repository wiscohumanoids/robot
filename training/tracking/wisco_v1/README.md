# Wisco V1 motion tracking

Human mocap (LAFAN1) -> Wisco joint trajectories -> mjlab tracking task `Wisco-V1-Tracking-Flat` (BeyondMimic-style,
whole body, 29 joints: every joint except the head). Each retargeted clip has a quality report beside it
(`motions/<clip>.report.json`). The Wisco robot config comes from `training/locomotion/wisco_v1/wisco_mjlab/`, the training model from `robots/wisco_v1/model/`.

| File | What |
|---|---|
| `retarget_lafan1.py` | LAFAN1 BVH -> Wisco CSV (own IK retargeter, numpy + MuJoCo) + quality report (+ optional preview video) |
| `compare_strip.py` | Same moments of a clip: Unitree's G1 retarget (top) vs our Wisco retarget (bottom) |
| `motions/<clip>.csv` | Retargeted clip, 30 fps: pelvis xyz, pelvis quat xyzw, 29 joints (training model order) |
| `motions/<clip>.report.json` | Quality numbers for that clip; `.debug.npz` per-frame errors |
| `wisco_tracking/` | The tracking task (copy of mjlab's G1 tracking config, Wisco names + Wisco randomization) |
| `train.py`, `play.py` | Launchers: register our tasks, run mjlab's own train / play |
| `train_wisco_tracking.sh <clip>` | VM job: CSV -> motion npz (`~/hsim_data/motions/wisco/`), then train |

Reproduce a clip (Mac):
```
# LAFAN1 BVH: lafan1.zip from github.com/ubisoft/ubisoft-laforge-animation-dataset (CC BY-NC-ND 4.0)
.venv-mjlab/bin/python training/tracking/wisco_v1/retarget_lafan1.py --bvh dance1_subject2.bvh \
    --out training/tracking/wisco_v1/motions/dance1_subject2.csv --video preview.mp4
.venv-mjlab/bin/python training/tracking/motion_to_npz.py --robot wisco --input training/tracking/wisco_v1/motions/dance1_subject2.csv \
    --output /tmp/wisco_dance1_subject2.npz --device cpu
CUDA_VISIBLE_DEVICES= .venv-mjlab/bin/python training/tracking/wisco_v1/train.py Wisco-V1-Tracking-Flat \
    --env.scene.num-envs 4 --agent.max-iterations 2 --agent.logger tensorboard --env.commands.motion.motion-file /tmp/wisco_dance1_subject2.npz
```
Train on the VM: `training/tracking/wisco_v1/train_wisco_tracking.sh dance1_subject2 --log-root ~/hsim_runs/wisco_track_dance1`.
Watch a trained policy: `tools/play_mjlab.py --task Wisco-V1-Tracking-Flat --policy <run>.onnx --motion-file <npz> --video out.mp4`.

If the retargeter changes, rerun it: the VM launcher reconverts the npz automatically when the CSV is newer.
