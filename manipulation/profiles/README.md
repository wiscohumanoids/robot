# Arm profiles

One YAML file per arm (or pair of arms) that a manipulation policy can be trained for. `manipulation/finetune.sh`
takes the profile name, and the trainers check data and models against it.

| Profile | Arm | Methods |
|---|---|---|
| [`panthera_sim`](panthera_sim.yaml) | one Panthera-HT arm in MuJoCo, block stacking (`panthera_sim/`) | `act`, `pi05` |
| [`panthera_bimanual`](panthera_bimanual.yaml) | the two real Panthera-HT arms, gear-and-pin assembly (`panthera_teleop/`) | `pi05` |
| [`g1_dex1`](g1_dex1.yaml) | the simulated G1 with Dex1 grippers at a counter (`training/manipulation/`) | `groot` |

## Fields

| Field | Meaning |
|---|---|
| `name`, `description` | the name used on the command line (= file name) and one line about the arm |
| `robot_type` | written into checkpoints; the deployment side compares it |
| `fps` | control and dataset rate, Hz |
| `state_dim`, `action_dim` | sizes of `observation.state` and `action` |
| `cameras` | dataset image keys, in the order the policy sees them |
| `image_size` | square image size the policy is trained at |
| `task` | the language instruction every episode must carry (`null`: per-episode instructions) |
| `simulation` | true for simulated arms: checkpoints record the simulator dynamics version |
| `methods` | the fine-tuning methods that apply (`act`, `pi05`, `groot`) |
| `pi05` | pi0.5 data options: `train_episodes` / `val_episodes` ([start, stop) ranges, checked), `normalization` (`train_episodes`: recompute state/action statistics on the training episodes; `checkpoint`: keep the checkpoint's on resume), `require_manifest` (a verified `panthera_teleop` export) |
| `groot` | GR00T options: `modality_config` |
| `deployment` | extra fields for each checkpoint's `deployment.json` (action representation, alignment, gripper convention, ...) |

## Adding an arm

1. Record and export a LeRobot dataset with the arm's cameras, state and action.
2. Add `<arm>.yaml` with the fields above and the methods that apply.
3. Train: `manipulation/finetune.sh <method> <arm> ...`. The checks stop the run if the dataset or model disagrees
   with the profile.
