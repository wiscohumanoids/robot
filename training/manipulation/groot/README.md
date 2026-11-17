# GR00T fine-tuning for standing pick-and-place (G1 + Dex1-1)

Pipeline: scripted + MimicGen episodes (`../sim`, raw npz) -> `raw_to_lerobot.py` -> GR00T LeRobot v2
dataset -> `finetune.sh` -> checkpoint -> `services/vla_server` -> `eval_closed_loop.py` / ROS `/skills/manipulate`.

| File | What |
|---|---|
| `g1_dex1_config.py` | GR00T modality config (NEW_EMBODIMENT): state arms 7+7, grippers 1+1, waist 3; actions arms (relative) + grippers (absolute), 16-step chunks; video `ego_view` |
| `raw_to_lerobot.py` | raw episodes (plan-doc format) -> LeRobot v2 + `meta/modality.json`, H.264 videos; keeps successful episodes only |
| `make_synthetic_raw.py` | fake episodes in the raw format, for pipeline tests only |
| `finetune.sh <version>` | fine-tune on `~/hsim_data/manip/<version>/lerobot` -> `~/hsim_runs/groot_finetune_<version>/` (N1.7 if the gated backbone is accessible, else N1.6) |
| `eval_closed_loop.py` | success rate in the manipulation env (../sim/env.py) through the VLA server (held-out seeds >= 1,000,000) |

## On the VM
```
# envs (done 2026-11-10): ~/hsim_envs/gr00t/Isaac-GR00T (N1.7, torch 2.9 cu128) and ~/hsim_envs/gr00t/Isaac-GR00T-n1d6 (N1.6, torch 2.7.1)
# weights: ~/hsim_data/hf-cache (HF_HOME); token via HF_TOKEN_PATH=~/.cache/huggingface/token
PY=~/hsim_envs/gr00t/Isaac-GR00T/.venv/bin/python
$PY raw_to_lerobot.py --raw ~/hsim_data/manip/v1/raw --out ~/hsim_data/manip/v1/lerobot
./finetune.sh v1                                    # 20k steps, batch 96; MAX_STEPS / BATCH / GROOT=n16|n17 to override
../../../services/vla_server/run_vla_server.sh      # newest checkpoint of groot_finetune_v1, tmux hsim-svc-vla, port 8200
MUJOCO_GL=egl <datagen env python> eval_closed_loop.py --episodes 50 --out ~/hsim_runs/groot_eval_v1 --video 5
```

## GR00T version
N1.7 (latest, Cosmos-Reason2-2B backbone) needs Hugging Face access to the gated `nvidia/Cosmos-Reason2-2B`
(auto-approved: open https://huggingface.co/nvidia/Cosmos-Reason2-2B logged in as the token's account and accept).
Until then `finetune.sh` uses N1.6 (ungated). A checkpoint must be served by the same version it was trained with
(`run_vla_server.sh` reads it from the run's `logs/run_info.txt`).
