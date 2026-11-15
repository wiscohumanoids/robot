# Human-in-the-loop collection

This adds human-gated intervention collection to the existing policy dashboard.
It follows [Dream Machines' intervention experiment](https://dream-machines.eu/blog/pi05-fine-tuning): run the policy, intervene before a mistake, then retain the
human correction segments and successful rollouts that needed no intervention.
Your control/camera rate and action representation stay at the existing Panthera
contract: 150 Hz target control, 20 Hz cameras/export, native 14D joint/gripper
values and 256 × 256 image crops. Recording runs at up to 150 state samples and
20 images per camera per second of **data time**. Policy playback continues to use bounded smooth
steps with a hold during replanning; this does not implement their asynchronous RTC.

## Try the controls without hardware

```bash
./scripts/start-inference-dashboard.sh --demo --port 8094
```

Open http://127.0.0.1:8094. The demo generates simulated arms, camera images and
policy predictions without connecting to robot devices or AWS. It writes to a
new temporary directory and marks recordings as simulation; the training selector
and exporter reject them. Hand guidance moves the simulated joints slowly.
Ctrl+C ends the demo.

## Collect live rollouts

Use the existing inference environment and AWS tunnel:

```bash
./scripts/inference-tunnel.sh
# In another terminal, with the existing recording controller stopped:
./scripts/start-inference-dashboard.sh --port 8093
```

Open http://127.0.0.1:8093. Port 8093 avoids the saved-run reviewer on 8091.
The default inference endpoint is http://127.0.0.1:8765; override `--url` if needed.
Raw recordings default to `data/hitl/`; `--data-dir` changes this directory.

1. With the reset path clear, **Connect & reset pose** and wait for hold.
2. Enter the operator and scene/setup names. Change the scene name when changing
   lighting, camera placement or object layout so you can track diversity.
3. **Play policy**. Adjust actions per step and blend using the existing motion
   controls if needed. **Play policy** automatically starts full rollout recording before motion, including
   unassisted runs. **Record full rollout** in **Session & playback settings** can
   still start recording manually. Finish and label each run before starting a new one.
4. When a mistake is approaching, press **H** or **Take over**. This cancels the
   pending inference/trajectory and holds both arms at their measured poses.
   The joints hold while grippers become passive. Hold still during baseline
   calibration (0.5 seconds), then pull a handle or open/close its gripper.
   The normal collector’s sustained-pull detector releases each arm independently
   into gravity assistance with a passive gripper (0.15-second confirmation).
   Gripper motion uses normal collection’s thresholds (0.03 radians of movement
   or 0.08 radians/second, sustained for 0.12 seconds). Recording starts when
   the first arm releases. Guide released arms by hand.
5. Press **P** or **End correction & hold** when finished. Both arms hold the
   measured pose. Release the handles, then **Play policy** to request a fresh
   prediction. Repeat the takeover/resume cycle as often as needed.
6. Use **Add notes / quality** if you want to rate the take or add a note. **K** keeps a successful episode, **R** saves a
   failed episode for review, and **D / Delete** moves it to recoverable discarded storage.
   Finishing holds the measured pose. Clear the path before using **Reset collection pose**
   to reset for the next rollout.

Keyboard shortcuts apply when a text/number field is not focused. **Esc** always
cancels motion and establishes position hold, including during hand guidance; it
leaves the rollout open so you can finish or continue. Closing the page sends
Stop; if heartbeats cease for two seconds, the active rollout is finalized as
incomplete Review. A camera/control/recording fault also prevents the rollout from
silently entering training. Reset is blocked during an active rollout.

Takeover is armed with the button/key even when full-rollout recording was not
started. Each arm’s joints stay held until its own sustained handle pull or gripper
movement; grippers are passive while waiting. Release remains
latched until End correction, Stop or a fault. Correction-only recording starts
on the first arm release, excluding both the policy lead-up and waiting hold.
Resumed policy sections may appear in the raw file but are excluded from correction
export. States include per-arm ownership, and metadata logs each handle release
with the handle/gripper trigger and measured baseline/change. Detection uses the same `neutral_return` hold
settings as normal collection. Keep the existing hardware stop accessible.
Live startup and reset use the existing robot lock, limits and trajectory checks.
Live Ctrl+C retains the existing powered-hold shutdown behavior; the reconnect
control releases connections only after its support confirmation.

### Active recording and safety slowdown

An open rollout pauses capture whenever both arms are in position hold, including
Stop, waiting for handle pulls, prediction previews and waiting for inference.
The rollout stays open; neither states nor camera images are saved during these
holds. One released arm is enough to record human guidance, including the held
arm's feedback. Human guidance uses real elapsed time while released.

During policy execution, capture advances with the safety-retimed trajectory's
policy clock. Each waypoint still spans 1/20 second of data time, even if speed or
acceleration limits stretch that waypoint in real time. For example, three real
seconds spent executing one policy second produce about one second of training
data. State and image saving slow together; the live control loop, cameras and
preview continue at their normal rates. The dashboard shows paused capture,
capture pace and accumulated data duration.

New recordings preserve physical receive times in `monotonic_ns` and
`episode_time_s`, and add `data_monotonic_ns` and `data_time_s` for training.
Camera data timestamps are mapped from their original receive times using the
same clock as states. `recording_segment` marks every restart or ownership change,
so two human corrections separated by an omitted hold remain separate episodes.
Metadata retains hold transitions without saving hold samples. The selector,
exporter and audits use the data clock when present and retain support for older
recordings that only have host timestamps. Existing recordings are not rewritten.

## Inspect and export training data

Every rollout contains `metadata.json`, `states.jsonl`, `frames.jsonl` and timestamped
JPEGs. Metadata includes checkpoint/contract, configuration/URDF/calibration snapshots,
operator, scene, outcome, quality, notes, control transitions and intervention count.
Each new state labels `human` or `policy` (older recordings also contain `hold`), alongside measured robot feedback,
the command target, current playback settings and host receive timestamps.
Frames preserve the host monotonic clock and share the active data clock; this is
not hardware exposure synchronization.

The selector reads only complete Kept rollouts. For assisted rollouts, it selects
each contiguous human correction as a separate training episode, excluding all
policy prefixes/suffixes. For unassisted rollouts, it requires marked success and
policy execution, trims idle ends, and keeps the active trajectory. Older files
retain their original wall-clock timing and intervening holds. It does
not apply a quality-score filter that would accidentally remove recovery behavior.
Segments shorter than 0.6 seconds are reported as skipped; the exporter still checks
camera coverage, alignment and minimum sample count. Review, discarded, interrupted
and simulation runs are excluded. Kept correction segments can be useful even if
the overall task failed: review the run and promote it to Kept before selection.
For the saved-run reviewer, use a separate annotations path so normal demonstration
review remains separate:

```bash
python3 scripts/review_runs.py --source data/hitl/review \
  --annotations config/hitl-review.json --port 8092
```

Use **Change category** to promote approved corrections to Kept, then run the
ownership selector below. Reviewer crop annotations are not used for HITL export;
shorten the generated per-correction intervals if further trimming is needed.

```bash
python3 scripts/prepare_hitl.py --output config/hitl-round1.json

work/postprocess-venv/bin/python scripts/postprocess_kept.py \
  --source data/hitl/kept --annotations config/hitl-round1.json \
  --output data/processed/hitl-round1-20hz \
  --trimmed-output data/processed/hitl-round1-full-rate \
  --repo-id FoxNerdSaysMoo/panthera-hitl-round1

work/postprocess-venv/bin/python scripts/verify_processed.py \
  data/processed/hitl-round1-20hz --source data/hitl/kept

work/postprocess-venv/bin/python scripts/verify_full_rate.py \
  data/processed/hitl-round1-20hz --source data/hitl/kept
```

Use new annotation/output paths for later rounds. `--template` selects an export
configuration; task, FPS, image size and crops must match the recorded inference
contract. Generated annotations use a unique segment ID plus `source_id` pointing
to the preserved original rollout. You can shorten intervals in the annotation
file after visual review. The exporter rejects corrections that contain policy or
hold states and rejects policy segments from an assisted rollout being presented
as unassisted. Each segment is a separate LeRobot episode; next-state targets and
LeRobot's terminal action padding do not cross a handoff.

Actions follow the existing measured-next-pose convention at t + 0.05 seconds of data time,
including successful policy rollouts. They are not the logged policy command
targets. Exported velocities also use data time; raw velocity feedback is preserved
in the recording. This keeps corrections compatible with the existing kinesthetic dataset.
Sources and all original images stay intact. The commands above create and audit
local artifacts; they do not upload data or start fine-tuning. Use the verified dataset alongside your compatible base demonstrations when
assembling the next fine-tuning round.

## Verification

```bash
work/teleop-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
work/postprocess-venv/bin/python -m unittest discover -s tests -p test_hitl_export.py -v
```

Handoff tests cover holding without force, independent latched release, recording
from the first release, held-arm tracking faults, actual SDK command routing with
mock robots, canceling in-flight inference, fresh resume, measured hold, reset
gating, browser loss, metadata and interruption quarantine. The export test
creates a saved rollout with two corrections, exports separate LeRobot episodes,
and audits images, actions, archives and native action-chunk boundary padding.
Live gravity/position handoffs remain to be checked at the robot with the operator.

### Finish and task-stage flags

Use the Succeeded / Failed buttons for gear pickup, transport, insertion, seating,
and pin pickup/insertion/pushing to label individual stages. Click a selected button again
to clear it; leave unattempted or unknown stages unmarked. These optional labels
are saved in `outcome.milestones`, separately from overall success and intervention
provenance, and do not add intermediate EXPO rewards.

K / R / D finishes the recording and then automatically resets both arms to the
collection pose using the existing bounded reset motion. Clear the return path
before finishing; Esc / Stop interrupts the reset. Restore the objects yourself
before the next Play. Reset motion is outside the recording.
