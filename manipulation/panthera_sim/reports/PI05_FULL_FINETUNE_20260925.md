# pi0.5 full fine-tuning notebook — 2026-09-25

Notebook: `notebooks/pi05_ik_three_block_full_finetune.ipynb`.
Target: one RTX PRO 6000, 96 GB VRAM. Defaults: batch 8, 30,000 optimizer
updates, bfloat16, gradient checkpointing, all parameters in AdamW, no PEFT.

## Published inputs

- Dataset: https://huggingface.co/datasets/FoxNerdSaysMoo/panthera-ik-three-block-stack-30hz
- 1,000 scripted IK three-block episodes; 862,766 training frames; 30 Hz.
- Both 256×256 shoulder/wrist RGB views are embedded in Parquet.
- Seven-dimensional state and absolute joint/gripper commands; existing
  next-uniform-sample alignment is preserved.
- Upload verified 258 data/metadata/card/audit files against local sizes and
  SHA-256 for LFS files. `upload_manifest.json` contains all source hashes.
- Data completion commit: `5bc24d17cba6dbaea531a80c3607cc3254473319`.
- Data plus simulator/runtime commit pinned by the notebook:
  `cc138c0ea9ec5332a7754ee1b1f5be4de75b4226`.
- LeRobot source: `e624f3f7f8411ec3a02635d06e79373341e5ef35`.
- Base: `lerobot/pi05_base` at `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`.

The runtime includes the current simulator/environment/scene/meshes, camera code,
and training/evaluation helpers. The simulator and scene hashes match the
demonstration provenance. Its manifest records runtime file hashes and every
collection seed. The notebook verifies hashes and checks benchmark seed disjointness.

## Training behavior

The wrapper uses upstream LeRobot training, evaluation loss, and resumable
checkpointing. It replaces the permissive pi05 weight loader with strict local
safetensors loading and checks all weights are trainable and present in the
optimizer. A first-update guard checks actual nonzero, finite gradients in the
vision tower, visual projector, language layers, action expert, and action output.
Unused language-generation heads remain optimizer-eligible; an action-only loss
does not necessarily produce gradients for every tensor on every batch.

Training episodes are 0–899; held-out loss uses 900–999. Numeric quantiles are
recomputed exclusively from training episodes. Processors are freshly built for
these features/statistics; resume and inference use saved checkpoint processors.

## Rollout reporting

Videos are H.264 MP4 at 30 simulated Hz, shoulder and wrist side by side.
Success requires green on the table, red on green, blue on red, adjacent XY
error <=12 mm, vertical gaps 32–60 mm, no active grasp, and 30 continuous stable
control steps. The original contact-triggered compliant-pad grasp assist is
retained. Only actuator-range clipping is applied and its frequency is reported.

The benchmark defaults to 50 fresh seeds for each of two start distributions:
collector-region starts without filtering for planner success, and the broader
teleop range. It reports those scores separately, with Wilson 95% intervals,
intermediate milestones, per-seed outcomes, errors included in the denominator,
and selectable saved rollout videos. Preview seeds are development seeds reused
in the benchmark; reserve another range for final testing after model selection.

## Validation performed locally

- Notebook schema, every code cell, and embedded subprocess Python parsed.
- Notebook training CLI parsed against the pinned upstream configuration.
- Full-size pi05 instantiated on PyTorch's meta device: **4,143,404,816** parameters.
  Actual base checkpoint tensor names/shapes matched the model after upstream
  remapping (no missing/unexpected tensors or shape mismatches).
- Pinned LeRobot loaded actual episode 999, both cameras, seven-dimensional state,
  and 50×7 action chunks at both episode boundaries.
- Four regression/integration tests passed in the pinned LeRobot CPU environment.
  The reduced-size real pi05 architecture completed two backward/optimizer steps
  with gradient checkpointing and nonzero gradients in all five component groups,
  saved/reloaded identical weights, and produced a finite seven-dimensional action.
  A deliberately incomplete checkpoint raised an error. Training-statistics tests
  excluded held-out values. Metric tests reject wrong order, floating stacks,
  held cubes, and insufficient/interrupted stability.
- A fresh physics replay of demonstration 999 passed the new success metric:
  2.43 seconds maximum stable hold, first success at 22.77 simulated seconds.
- Stationary-policy rollouts failed correctly for both reset distributions.
  Both generated MP4s decoded as 512×256 H.264 at 30 FPS.

**Not performed:** a full-size pi05 forward/backward run or learned-policy success
benchmark. The local GPU has 8 GB VRAM. The notebook's actual four-update full-model
GPU smoke run and peak-memory report execute on the user's 96 GB GPU before the
long training run. No trained pi05 checkpoint or model success rate is claimed.

## Storage and W&B update

The original 300 GiB free-space hard gate was excessive: it budgeted for retaining
all periodic optimizer checkpoints. It has been removed. Startup prints free
space with an advisory; the smoke run saves diagnostics without weights/optimizer.
Training keeps the latest two completed checkpoints by default (configurable to
one), saves a replacement before pruning, and measures actual checkpoint tensor
sizes before each save. A failed save never triggers pruning. Only completed,
marked numeric checkpoints within the current run are pruned. Existing unmarked
checkpoints, other experiments, symlinks, and incomplete checkpoints are preserved.
The rolling save still requires temporary room for one additional checkpoint.

W&B is now enabled by default for the main run with model artifact uploads
disabled. Training logs include offline validation loss; resume uses the stored
run ID, or starts a logging run if the checkpoint predates W&B. Evaluation creates
grouped runs containing per-distribution success rates, 95% intervals, intermediate
milestones, errors, and optionally saved videos. API keys are passed securely in
the child process environment. Offline mode is supported; smoke runs remain unlogged.

Update validation: six regression/integration tests passed. The notebook CLI and W&B training/evaluation metrics plus H.264 video logging passed an offline integration test; no remote W&B run was created during validation.

## Runtime assets and Google Drive update

The earlier bundle included visual meshes but accidentally omitted collision
meshes. Previous simulator smoke checks used the full local checkout, concealing
the packaging error. The bundler now includes the complete mesh tree and compiles
the simulation in an isolated subprocess from a clean archive extraction before
publication. The regression was reproduced using the old archive. The corrected
archive compiled all 20 meshes and rendered both cameras from its isolated tree.

The notebook now mounts Google Drive and checks write access before installation,
model/data downloads, or GPU training. Each complete training checkpoint is copied
to `DRIVE_ROOT/RUN_NAME/<step>` with model, processors, deployment contract, and
optimizer/RNG state. The previous complete backup remains until the replacement
and manifest are written, then `latest.json` is updated and the older managed
backup removed. Other Drive folders are preserved. Failures preserve the local
checkpoint and prior Drive backup, and stop with an error rather than silently
losing persistence. The smoke run saves no checkpoint and performs no backup.

`RESUME_FROM_DRIVE=True` restores the latest backup to local storage and verifies
all file sizes and SHA-256 hashes before resuming. Ten tests passed, including
isolated asset loading, rolling backups, interruption preservation, and corrupted
restore rejection. Google Drive's actual authorization and mounted-filesystem
behavior must run in the user's Colab session; local tests use a temporary
filesystem, and no access to the user's Drive was attempted during development.
