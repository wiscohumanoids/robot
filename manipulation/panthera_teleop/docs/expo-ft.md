# Panthera EXPO-FT

This is an experimental PyTorch adaptation of [EXPO-FT](https://pd-perry.github.io/expo-ft/)
for Panthera's existing LeRobot pi0.5 policy. It includes a bounded stochastic edit
actor, ten Q networks, target-Q selection among eight original and eight edited
proposals, replay validation, exact decision recording, a separate inference
service, and optional successful-only pi0.5 flow-matching updates.

**No robot-trained EXPO policy has been produced yet.** The laptop training check
uses a labeled synthetic control problem. Existing Panthera demonstrations and
HITL recordings cannot be silently converted into faithful RL transitions: the
old exports use measured-next-pose actions, and the old recordings lack exact
decision observations/candidates. Fresh operator-supervised collection is needed.

## Research and implementation choices

The research reference is upstream commit
`21fe3d3b7d913c80836496817965932f49e0aedf` of
[pd-perry/expo-ft](https://github.com/pd-perry/expo-ft), alongside
[the paper](https://arxiv.org/html/2605.25477v2). The upstream implementation uses
JAX/OpenPI and DROID; this implementation retains Panthera's PyTorch/LeRobot,
native joint actions, local control checks, and existing saved processors.

For an executed prefix of C waypoints, the critic target is
`sum(gamma**i * reward[i]) + gamma**C * nonterminal * min(target_Qs)`.
The editor maximizes mean Q with an entropy term. Its bounded edit is
`beta * tanh(Gaussian)`, including the scale and tanh corrections to log density.
Both unedited and edited candidates participate in target-Q selection. Failures
belong in critic replay; success-only filtering applies to base supervision.

The local collection proxy samples one base candidate online; the offline cache
samples eight genuine candidates at each exact observation for learning. The
collection/serving default is C=17 consecutive waypoints, each with 14 absolute
joint/gripper actions, and the existing three 256x256 RGB crops at 20 Hz policy
time. EXPO collection sets blending to zero and rejects a different prefix or
blend setting. Each waypoint still goes through local smooth retiming, speed,
acceleration, joint-limit, tracking, freshness, heartbeat, and takeover checks.
Gripper projection is applied consistently to action selection and learning.
Discounting is per policy waypoint, not per second of elapsed wall time; real
timestamps and emitted commands remain in the ledger.

To fit the laptop, Q/editor observations use the checkpoint-normalized 14D state
and deterministic RGB block means (three 16x16 views, 2,318 total features).
This is a compact visual representation, **not the paper's visual encoder**;
small gear/pin details may require a stronger encoder after initial validation.
The base pi0.5 retains its native images and saved quantile processors.

`train_expo.py` trains the editor/critics with a frozen base and cached genuine
proposals. This is the residual-policy variant of EXPO. The separate base-training
command provides the original flow-matching component of full EXPO-FT; it is not
an automatic asynchronous online loop. Changing base weights invalidates proposal
caches and existing editor/base bindings: regenerate candidates and train a
matching bundle before deployment. UTD defaults to 20 critic updates per editor
update, reusing the supplied minibatch; this differs from upstream's partitioned
minibatches. Checkpoints include optimizers, targets, temperature, and RNG.

## Compute and local verification

At the October 5 inspection, the laptop's RTX 4060 had 8 GiB VRAM. AWS's single
96 GiB GPU was actively running `panthera-pi05-new50.service` and the existing
inference service. No AWS process, service, environment, or checkpoint was changed.
All new GPU entry points refuse another compute process on the selected GPU,
even if free VRAM remains. The lock coordinates these EXPO tools; it cannot stop
an unrelated application from starting later.

Create the local learner environment and inspect resources:

```bash
bash scripts/setup-expo-env.sh
work/expo-venv/bin/python scripts/expo-preflight.py --dry-run
work/expo-venv/bin/python -m unittest discover -s tests -p 'test_expo_*.py' -v
```

The setup shares installed packages from `RobotArmLearning/.venv-act` through a
new environment, without installing into that source environment. Its versions
are recorded in `work/expo-venv/expo-source.json`. `EXPO_SOURCE_ENV` can select
another compatible source. For an independent learner environment, use
`requirements-expo.txt` and a suitable CUDA PyTorch installation.

Run the hardware-free training check into a **new** output directory:

```bash
work/expo-venv/bin/python scripts/train_expo.py --synthetic \
  --output work/expo-runs/synthetic-check --batch-size 256 --steps 100 --utd-ratio 20
```

The completed local batch-256 check used about 0.30 GiB peak reserved VRAM, with
100 editor updates and 2,000 critic updates. Its synthetic reward improved from
approximately -0.368 to -0.109. This measures algorithm/compute behavior only;
it says nothing about real task success. Synthetic run checkpoints are rejected
by the real inference service. See `docs/expo-validation.json` for the recorded
verification results.

## Fresh rollout collection and learning rounds

The pi0.5 runtime needs the clean editable LeRobot commit
`e624f3f7f8411ec3a02635d06e79373341e5ef35`, its existing dependencies, and a complete
local Panthera checkpoint. The lightweight laptop environment is deliberately
insufficient for loading the full VLA. The saved model alone is larger than the
laptop's VRAM. Run VLA sampling/base updates on an idle capable GPU after current
jobs finish, or another isolated machine. Do not stop the existing AWS run to make
room. The new runtime also refuses the still-running inference process; plan a
separate idle GPU or a later operator-controlled transition.

1. On that idle GPU machine, in the pinned LeRobot environment with this project
   available, start the separate base-only collection service using a fixed
   checkpoint directory. It binds loopback port 8766, and refuses port 8765:

   ```bash
   python -m infra.rl.server --checkpoint /absolute/fixed/pretrained_model \
     --prefix-length 17 --port 8766
   ```

   Forward port 8766 through SSH if remote. The existing inference deployment
   scripts and production service are not changed by these commands.

2. On the laptop, after the normal operator setup and with the existing recording
   controller stopped, collect using the optional ledger:

   ```bash
   ./scripts/start-inference-dashboard.sh --url http://127.0.0.1:8766 \
     --expo-record-dir data/expo-c17 --expo-objective gear_insertion \
     --expo-prefix-length 17 --port 8094
   ```

   Start **Record trial** before Play. Label both successes and failures
   explicitly. Finish while waiting for inference safely cancels the unexecuted
   request and assigns the outcome to the last completed waypoint. Finishing
   during execution of a waypoint leaves an incomplete transition, which is
   excluded. Single-step playback also provides an explicit completion boundary.
   H/P/Stop retain their usual behavior; takeovers, cancellation,
   incomplete episodes and faults make that episode ineligible for the initial
   strict RL replay. Human corrections remain useful in the separate HITL/SFT
   workflow. In EXPO mode, **Failed trial (R)** explicitly approves a complete
   failed trial for replay; the interruption and provenance checks still apply.
   No command above starts physical motion without
   the dashboard's existing operator actions.

3. Cache genuine candidates at **both** exact current and next observations.
   With the existing frozen AWS policy reached through its loopback SSH tunnel,
   keep the ledger sessions on the laptop and run:

   ```bash
   python scripts/cache_expo_candidates.py \
     --session /absolute/data/expo-c17/SESSION_ID \
     --backend http://127.0.0.1:8765 \
     --manifest work/hitl-round1/remote-base-manifest.json \
     --output /absolute/expo-cache/round1
   ```

   This uses eight sequential backend requests per observation and loads no
   additional VLA. It sends the exact saved state/JPEG bytes, preserves each
   source gripper-limit correction, and checks the attested checkpoint identity.
   The backend controls its own sampling noise: remote mode rejects `--seed`
   and records `seed_controlled: false` in the cache.

   To use a separate idle machine with a local immutable policy instead:

   ```bash
   python scripts/cache_expo_candidates.py \
     --session /absolute/data/expo-c17/SESSION_ID \
     --checkpoint /absolute/fixed/pretrained_model \
     --output /absolute/expo-cache/round1
   ```

   Repeat `--session` for additional sessions. This preserves raw ledgers and
   saves `normalization.json`, a checksummed candidate file, and a cache manifest.
   Canceled rows are skipped. Cache identity binds exact state/JPEG input,
   checkpoint, and processors. Do not replace candidates with recorded actions.
   Cached proposals remain physical radians; replay preparation applies the
   saved checkpoint normalization exactly once. The cache retains the genuine
   full 50-waypoint proposals; preparation selects their first 17 waypoints.

4. Copy the cache back to the laptop, build replay, and train:

   ```bash
   work/expo-venv/bin/python scripts/prepare_expo_replay.py prepare \
     --ledger-root data/expo-c17 --normalization /absolute/expo-cache/round1/normalization.json \
     --candidate-cache /absolute/expo-cache/round1 --training-objective gear_insertion \
     --prefix-length 17 --output data/expo-replay/round1-c17.npz

   work/expo-venv/bin/python scripts/train_expo.py \
     --replay data/expo-replay/round1-c17.npz --output work/expo-runs/round1-c17 \
     --prefix-length 17 --batch-size 256 --steps 1000
   ```

   Sparse reward is one only on an explicitly successful terminal waypoint;
   all other rewards are zero. Complete failures are retained. All transitions
   from an original episode stay in one split; held-out sources are excluded
   from training. Output directories and replay paths must be new.
   Every decision must complete all 17 waypoints. Partial execution is excluded,
   never padded or repeated. Continuation uses `gamma**17`, and successful
   terminal credit is at waypoint 17 (discounted by `gamma**16`). Every session
   in an export must declare the same chunk length. Earlier C=1 sessions remain
   in `data/expo`; prepare and train those separately with explicit
   `--prefix-length 1`.

   For `gear_insertion`, success means the gear is fully seated in the carrier
   and stays seated after the policy releases it;
   no pin is required. Replay transitions, `run.json`, and the trained checkpoint
   carry this reward objective. `gear_and_pin` episodes cannot share its replay
   or critic training run. This changes reward interpretation only; the frozen
   pi0.5 checkpoint and its saved base task prompt stay unchanged.

5. On an idle VLA GPU, load the trained editor alongside the **same** base using
   `--rl-checkpoint /absolute/round1-c17/checkpoint-001000.pt --prefix-length 17`. Keep its sibling
   `run.json`; the server verifies real-data eligibility, normalization, base
   identity, dimensions and trained reward objective. `--explore` enables stochastic edits during collection;
   default inference uses deterministic edits for evaluation. Repeat collection,
   candidate caching, replay preparation and training as separate supervised rounds.

The service and collection path have been tested with injected policies and
hardware-free fixtures. Real pi0.5 sampling and robot EXPO execution remain
unverified in this workspace because the available VLA GPU is occupied.

## Data audit and approval

```bash
work/expo-venv/bin/python scripts/prepare_expo_replay.py audit \
  --output work/expo-data-audit-next.json
```

The initial audit found 277 unique processed demonstration sources and 10 HITL
recordings, but zero faithful Q transitions. Processed releases overlap; combining
all release directories would duplicate source demonstrations. A kept label is
not proof of task success. Assisted success cannot reward the policy prefix.

The experimental ledger is append-only during recording and uses separate
`manifest.json`, `episodes.jsonl`, `decisions.jsonl`, `commands.jsonl`, and exact
JPEG files. The initial builder accepts complete unassisted Kept episodes or
Review episodes whose explicit ending has `review_approved: true`. For previously
unapproved Review recordings, preserve the original session and approve only a
reviewed working copy after checking its explicit outcome. In the EXPO dashboard,
**Failed trial (R)** records `success: false` and
`review_approved: true` as the operator's explicit failure label. This approval is
permitted only for failed EXPO trials. Discarded, assisted and incomplete sources
stay excluded. Ordinary HITL Review keeps its existing behavior.

## Operator protocol: refine gear insertion

The C=17 gear-insertion collector uses the local proxy on port8767 and a
separate dashboard on port8094:

```bash
./scripts/start-inference-dashboard.sh --url http://127.0.0.1:8767 \
  --expo-record-dir data/expo-c17 --expo-objective gear_insertion \
  --expo-prefix-length 17 --port 8094
```

Start each trial from the configured reset/collection pose, with the loose gear
and carrier arranged in the usual task starting scene. Complete any manual scene
setup before **Record trial** and release the handles before **Play**. The policy
performs the entire first motion: gear pickup, transport, insertion and release.
Recording fixes 17 waypoints per decision and zero blending. Use a proxy started
with `--prefix-length 17`; the dashboard checks that the server advertises the
same prefix. Keep earlier C=1 sessions and any running trial separate.

Press **Record trial**, then **Play**. Press **K / Gear seated** only when the
gear is fully seated and stays seated after the policy releases it. Finish before
pin handling; this objective does not require a pin. Press **R / Failed trial**
for a complete, clearly unsuccessful unassisted attempt. Press **D / Discard**
for partial or visually unverified outcomes. Finish while waiting for the next
prediction; finishing during a partly executed waypoint excludes that trial.
Human help, Stop, faults and interruptions also exclude the trial from strict
replay. Reset the robot and restore the loose gear starting scene before the next
trial; perform that setup before recording.

The session, episode and replay declare `training_objective: gear_insertion`.
Its sparse terminal reward trains the critic/editor for insertion while the
pi0.5 weights and original task prompt remain fixed. There are no approach,
grasp or other intermediate rewards. Gear-only and full-assembly objectives use
separate replay and trained bundles.

The legacy `--expo-objective gear_and_pin` mode requires actual gear-and-pin
assembly success for **K / Full success**. Gear insertion alone or an approach to
the pin hole does not satisfy that full-assembly outcome.

## Optional base pi0.5 update

The current operator workflow is EXPO only with frozen pi0.5 weights. Base updates
are not enabled. The following is a separate optional implementation path.

The full-model component uses a reviewed full-horizon supervision NPZ, separate
from C=17 critic replay. Fields are `action[B,50,14]`, `observation.state[B,14]`,
the three `observation.images.<camera>[B,3,256,256]` arrays, `task[B]`,
`episode_success[B]` booleans, `supervision_valid[B,50]` booleans,
`action_provenance[B]`, and `source_episode_id[B]`. Images may be uint8 or float
RGB in [0,1]. Padded or incomplete sequences are excluded. The schema check does
not verify correspondence to original recordings; that requires prior review.

On an idle GPU in the pinned LeRobot environment:

```bash
python -m infra.rl.base_training \
  --checkpoint /absolute/fixed/pretrained_model \
  --batch /absolute/reviewed-success-full50.npz \
  --output /absolute/expo-base/round1 --batch-size 8 --steps 100
```

This uses original pi0.5 flow loss and retains saved normalization. It saves a new
base checkpoint without modifying the source. It is a separate supervised stage,
not a replacement for Q/editor reinforcement learning. Regenerate candidate
caches and use a new matching RL run after this stage. Full-model batch size 8 is
an example, not a measured capacity claim; the idle-GPU resource gate still applies.
