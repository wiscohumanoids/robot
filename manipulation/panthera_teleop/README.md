# Panthera gravity assist and recording

## Getting started

On an already configured machine, run `./start.sh` from the repository root. A fresh clone also needs the manufacturer's [Panthera-HT SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK) and a Python environment: the local `work/` folder is excluded from Git.

### 1. Install the software

The commands below use Linux x86_64, Python 3.10, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The vendor's prebuilt motor wheel targets Linux x86_64 and was built on Ubuntu 22. For other platforms or binary compatibility issues, follow the [SDK's source-build instructions](https://github.com/HighTorque-Robotics/Panthera-HT_SDK/blob/4a267c148cffa56b369222110bcd0b925809e365/panthera_python/README.md).

```bash
git clone https://github.com/zebulontaylor/Panthera-Teleop.git
cd Panthera-Teleop
mkdir -p work
git clone https://github.com/HighTorque-Robotics/Panthera-HT_SDK.git work/Panthera-HT_SDK
git -C work/Panthera-HT_SDK checkout 4a267c148cffa56b369222110bcd0b925809e365

uv venv --python 3.10 work/teleop-venv
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python work/teleop-venv/bin/python \
  work/Panthera-HT_SDK/panthera_python/motor_whl/hightorque_robot-1.2.0-cp310-cp310-linux_x86_64.whl \
  -r work/Panthera-HT_SDK/panthera_python/requirements.txt \
  -r requirements-recording.txt 'pin==4.1.0'
```

The SDK revision and Pinocchio version above match this setup. `UV_SKIP_WHEEL_FILENAME_CHECK=1` accommodates the vendor wheel's filename/metadata version mismatch. The launchers use `work/teleop-venv/bin/python` directly, so activating the environment is optional.

Check imports, then try the dashboard in simulation:

```bash
work/teleop-venv/bin/python -c 'import hightorque_robot, pinocchio, numpy, yaml, cv2'
./start.sh --demo --data-dir /tmp/panthera-demo
```

Simulation opens the dashboard at http://127.0.0.1:8090 without opening motor devices. Stop it with Ctrl+C before starting live control.

### 2. Configure your hardware

Live recording uses two Panthera arms and three USB cameras. Connect the arms through their communication sockets, power their motor supplies, and enable motor power with the round buttons. Your user must have access to the robot serial devices and camera devices.

The checked-in USB mapping and camera paths describe the original laptop. On another machine, update:

- `config/robot-configs/arm-mapping.json`: set each arm's `usb_path` and `pci_controller` from its Linux USB topology. Inspect `readlink -f /sys/class/tty/ttyACM*/device` and identify which physical arm is left/right. The tested boards share a serial number, so serial-number links cannot distinguish them.
- `config/recording.json`: set the three `cameras` paths using `ls -l /dev/v4l/by-path/` and confirm each view. Close other applications using the cameras.

Check the mappings and URDFs without opening motor devices:

```bash
./scripts/start-hand-guide.sh
```

This regenerates the left/right robot YAMLs, including absolute SDK/model paths, for your checkout. Keep the robot cables in the configured hub ports afterward.

### 3. Start recording

With both arms supported and the return path clear, run:

```bash
./start.sh
```

The dashboard opens at http://127.0.0.1:8090. Guide the arms by their physical handles; K keeps an episode, R flags it for review, and D discards it. Each decision triggers a raised-neutral return and gripper opening. Support both arms before Ctrl+C ends assistance. Recordings are saved under `data/` and are excluded from Git. The recorder reserves 2 GiB of free disk space.

For optional Quest controller operation, also clone [Quest_controller_stream](https://github.com/HighTorque-Robotics/Quest_controller_stream) into `work/Quest_controller_stream`, install its Python/UI dependencies into the same environment, and install Android `adb`. Follow [the Quest guide](docs/quest-quick-start.md); the preserved passthrough changes are in `archive/setup/quest-passthrough.patch`. Physical handle mode uses the Panthera SDK without a headset or Quest bridge.

## Recording and control

Run `./start.sh` from this workspace. It checks all three cameras and both arm configurations, enables independent gravity assistance, and opens http://127.0.0.1:8090. Grippers are passive during demonstrations and open to 60% during each reset. There is no starting-pose or exit trajectory. Episode decisions return both arms to a raised neutral pose. The session started by Codex runs as the user service `panthera-gravity-assist`; inspect it with `systemctl --user status panthera-gravity-assist` and support both arms before running `./scripts/stop-gravity-assist.sh` to stop it. A running session must be stopped before launching another.

Move either arm to start an episode automatically. The recorder includes 0.75 seconds of pre-roll and requires sustained motion for 0.12 seconds to reject encoder jitter. Keep the browser page focused:

| Key | Action |
| --- | --- |
| K | Finish and keep the episode |
| R | Finish and flag the episode for review |
| D | Finish and move the episode to recoverable trash |

Release the handles and clear the return path before pressing K/R/D. Each decision immediately starts returning both arms to `[0, 0.25, 0.25, 0, 0, 0]` radians. The model places the tool about 6.4 cm above the zero pose. Joint speed is capped at 0.60 rad/s (2× the preceding speed setting), with a 0.6-second minimum trajectory duration, a 2.0 rad/s² acceleration cap, and return torque caps are `[5, 8, 8, 3, 2, 2]` Nm. Recording stays paused throughout the return and clears its pre-roll afterward. Arrival requires both arms within 0.015 rad of target and no more than 0.004 rad of encoder motion across a 0.25-second window. Reported velocity spikes cannot reset arrival while the encoder pose is stable. Real positional error or vibration still prevents arrival. Return logs include joint error, reported speed, encoder motion, and planned duration. Both arms remain in powered position hold at neutral. Grab a handle and gently pull: a sustained change in measured motor torque releases that arm into gravity assistance, independently of the other arm. A 0.5-second hold baseline and 0.15-second confirmation reject brief torque spikes. Pull thresholds are configured per joint as `[0.6, 1.2, 1.2, 0.6, 0.25, 0.25]` Nm. Recording remains blocked while both arms are held; the first released arm rearms it and movement starts the next episode. The other arm stays held until its own handle is pulled. During the return, both grippers move to 60% of their configured travel (currently 1.2 rad) at 0.8 rad/s with a 0.5 Nm torque cap. Each becomes passive once it reaches the reset opening, so it can be closed freely. Pulling an arm handle also cancels any remaining gripper opening for that arm. An obstructed gripper is released to passive control after 5 seconds and its error is shown on the dashboard; the arms keep holding. Gripper reset outcomes are included in the return log. Return results are saved in `data/return-events.jsonl`. A rejected path, tracking deviation, or arrival timeout cancels the return, shows an error, and holds the measured poses until handle pulls release them. Joint/motor/camera faults still stop the session. The tool path is checked against dipping below its endpoint heights; this is not obstacle or collision detection. Idle pauses during a demonstration do not end it. Support both arms before stopping: Ctrl+C in the controller terminal ends assistance and sends the SDK stop command. Closing the dashboard does not stop the controller.

Episodes live under `data/kept`, `data/review`, or `data/discarded`. Discarded episodes remain recoverable and are excluded from the kept dataset. An unfinished episode goes to review on shutdown. After an abrupt interruption, pending episodes are recovered to review on the next launch. No test recordings enter these directories; simulation defaults should be directed to a temporary directory.

Each episode contains `metadata.json`, `states.jsonl`, `frames.jsonl`, and timestamped JPEGs in `images/<camera>/`. States include both arms' positions, velocities, measured torques, gripper states, motor modes/faults and SDK feedback times; sent gravity torques and zero-stiffness commands; URDF-derived tool poses; host wall-clock and monotonic timestamps; sequence numbers and control compute time. Metadata snapshots the URDFs, robot and motor configurations, USB mapping, thresholds, camera settings, and calibration status. Camera frames carry individual timestamps, indices, actual image sizes, and file paths. Match camera and joint observations by their shared host monotonic clock. These are receive timestamps, not hardware exposure synchronization.

Control targets 150 Hz; actual achieved rate can be measured from the state timestamps. Cameras target 20 Hz at 640 × 480 with JPEG quality 85. Recording I/O runs on a separate thread with a bounded queue. The session stops and preserves its active episode for review on recording failure, lost camera frames, invalid/stale motor feedback, motor faults, or joint-limit violations. Two GiB of free disk space is reserved. Joint limits and the existing gravity torque caps are retained. Camera intrinsics, hand-eye transforms, and wrist-to-arm assignments remain unverified; each tool pose is expressed in that arm's model base frame.

The folders are organized by purpose:

- `app/`: controller, recorder, and local browser dashboard.
- `config/`: USB mapping, active robot YAMLs, camera and recording settings, calibration materials.
- `scripts/`: alternate Quest launcher, headset setup, configuration-only hand-guide launcher, and diagnostic probe.
- `data/`: classified episodes; `logs/`: runtime logs and the controller lock.
- `tests/`: hardware-free episode and control checks.
- `docs/`: setup history and Quest instructions.
- `archive/`: earlier experiments, result JSONs, camera captures, old board configs, and setup downloads.
- `work/`: vendor SDK, Quest bridge, Python environment, and Android platform tools. Keep these dependencies in place.

Check configuration without enabling motors: `./scripts/start-hand-guide.sh`.

Test without motors: `./start.sh --demo --data-dir /tmp/panthera-demo`; add `--demo-cameras` to include real camera capture. Simulation is clearly marked in the dashboard and metadata.

Run hardware-free checks: `work/teleop-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`.

Never run a separate probe or legacy experiment while control is active. The current gravity and Quest launchers share an exclusive controller lock. Keep robot USB cables in the saved hub ports.

If the serial connection drops during an idle hold, install the targeted USB protection once, with the controller stopped:

```bash
sudo ./scripts/enable-robot-usb.sh
```

Reconnect both robot USB cables afterward, then launch normally. This disables autosuspend for the Livelybot boards and the tested Terminus robot hub, and excludes the boards' serial ports from ModemManager probing. At startup the hand-guidance controller also keeps each board's upstream USB ancestry awake when permissions allow; otherwise it prints the setup command. These settings mitigate host power management and probing; they cannot prevent cable, supply, or firmware resets. The September 30 failure coincided with kernel USB disconnect/re-enumeration events for both boards, while control commands were still being sent.

A recognized serial disconnect now ends control cleanly with exit status 1, attempts to stop both arms, and finalizes any unfinished episode for review, without a Python traceback. Support the arms and restart after checking the USB connection. Control does not automatically resume through the SDK's background reconnect path, which clears motor objects referenced by the Python wrapper.

If a board appears in `lsusb` but no `/dev/ttyACM*` port appears, startup distinguishes an unconfigured USB device from a missing board. Kernel messages such as `rejected 1 configuration due to insufficient available bus power` mean Linux refused to configure the device. Check the hub's external power connection or use a powered hub, then reconnect it. Autosuspend settings do not fix a missing USB power budget.

## Postprocessed gear/carrier/pin demonstrations

**Camera setup disclaimer (October 2, 2026, America/Chicago):** The overhead camera was repositioned on this date. Recordings collected after the change have a different overhead viewpoint from earlier recordings, so the visual data are not identical across collection sessions. Future Hugging Face releases that include these recordings will contain both camera setups. The existing September 29–30 export predates this change. Use the original episode collection timestamps in `processing/source_metadata/` to identify the collection date; the date alone does not identify whether an October 2 episode was recorded before or after the camera move.

The September 29 and 30 sessions contain 102 kept demonstrations. Their combined
export is under `data/processed/panthera-gear-carrier-pin-real-20hz-102/`.
The first 30 episodes were reused from
`data/processed/panthera-gear-carrier-pin-real-20hz/`; only the 72 new episodes
were transformed, under `data/processed/panthera-gear-carrier-pin-sept30-20hz/`.
Full-rate trimmed copies are under `data/processed/trimmed-full-rate/` and
`data/processed/trimmed-full-rate-sept30/`, and every episode also has an archive
in the combined export. Original kept recordings are preserved.
The export uses LeRobot v3.0, 20 Hz, three 256 × 256 RGB views,
and native 14D bimanual joint/gripper states and next-pose actions. Images use
fixed square crops without padding; the overhead view keeps a tight 256 × 256
work area at native pixel resolution. `config/postprocessing-gear-pin.json`
records the original 30 visual release cutoffs, crop coordinates and timing/action
contract; `config/postprocessing-gear-pin-sept30.json` annotates the new 72.

Rebuild into new output directories with `scripts/postprocess_kept.py` and audit
with `scripts/verify_processed.py` and `scripts/verify_full_rate.py`, using
`work/postprocess-venv/bin/python`. `scripts/merge_processed.py` combines compatible
processed batches while checking source identity and preserving embedded JPEG
bytes. `scripts/publish_processed.py --update-existing <combined-export>` updates
the existing Hugging Face repo after verification, preserving a snapshot tag of
the previous release.
The dataset README contains the full commands, source limitations and inference
crop requirements. These scripts only process saved files and never send motor
commands. The Hugging Face destination is
https://huggingface.co/datasets/FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz.

## Trim recovery clips from flagged runs

Run `python3 scripts/review_runs.py`, then open http://127.0.0.1:8091.
This standalone tool reads saved recordings only. It does not connect to the
arms or cameras, and can run alongside the recording dashboard.

The queue includes all Kept recordings as well as Review recordings, while
preserving their categories. Use the category filter to show only Kept runs.
Choose a run from the review queue and drag across the selection timeline to
keep the recovery. Drag either handle to refine the start or end; the camera
views follow the boundary as you drag. Click the timeline to seek without changing
the selection. Focus a handle and use arrow keys for 0.05-second adjustments
(Shift for 1 second). You can also set **Keep from** and **Keep until** using
the playhead buttons or exact seconds. The three camera
views use the shared host recording clock. Arrow keys step overhead camera frames;
Space toggles playback, I/O mark boundaries, and S saves. Slow playback and
**Preview kept range** help check the cut. One contiguous interval is retained
per run. Use **Save & next unedited** to work through the queue. Saved badges,
a progress count, and the Edited / Not edited filter track work across restarts.
Switch runs freely without a warning; unfinished selections and note text stay
in memory while the page is open. Use **Save notes only** to persist notes without
approving a crop or marking the run edited. Notes live in the companion
`config/postprocessing-review.review-state.json` file. **Change category** moves
the original recording between Review, Kept, and Discarded and updates its label.
Moved runs remain accessible in the queue, including through category filters.
Discarding removes the run from the crop export list; moving it back does not
automatically approve it again.

Edits are atomically saved to `config/postprocessing-review.json` in the same
annotation format as the regular postprocessor. Only saved runs appear in that
file; crop saves leave original recordings and their labels unchanged. Category
changes update their folder and label; saved crops track that source category. The default
processing settings and image crops come from
`config/postprocessing-gear-pin-sept30.json`. Override `--template`, `--source`,
`--annotations`, or `--port` for a different task or recording folder.

Export the saved selections using new output directories:

```bash
work/postprocess-venv/bin/python scripts/postprocess_kept.py \
  --source data/review \
  --annotations config/postprocessing-review.json \
  --output data/processed/review-recoveries-20hz \
  --trimmed-output data/processed/review-recoveries-full-rate \
  --repo-id FoxNerdSaysMoo/panthera-review-recoveries
```

The exporter requires explicit `review_approved: true` annotations for review
runs, which the editor adds on save. It retains the normal complete-recording,
alignment, camera-coverage, and minimum-sample checks. An interrupted recording
is shown with a warning in the editor and is not eligible for this exporter
without separate source validation. Exporting does not upload or publish data.

## Human-in-the-loop policy corrections

The policy dashboard now supports recorded rollouts with **H** to take over into
hand guidance and automatically start recording at takeover, **P** to end a
correction and hold, then **Play** to resume with a fresh prediction. Recording a
full rollout before Play is optional. Keep/review/discard outcomes include quality, notes, operator
and scene. Export selects human correction segments plus successful unassisted
rollouts, with separate episode boundaries for every correction.

Try `./scripts/start-inference-dashboard.sh --demo --port 8094` without hardware.
For live collection, run the existing inference tunnel and launch the same command
without `--demo`; recordings go to `data/hitl/`. See the
[collection and export guide](docs/human-in-the-loop.md) for the full workflow.

## Experimental EXPO-FT reinforcement learning

The [EXPO-FT guide](docs/expo-ft.md) covers the new lightweight edit actor/Q
learner, exact policy-decision recording, genuine pi0.5 candidate caches, and
separate guarded inference/training commands. The laptop passed a batch-256
synthetic training check. Real Panthera RL still needs fresh eligible rollouts;
the existing SFT exports cannot substitute for executed-action replay.
