# AWS policy inference

AWS: `admin@3.83.15.102` (existing SSH config/key). The pinned LeRobot runtime
already lives at `/mnt/pi05/lerobot/.venv`; deployment does not replace it.
The selected model is pi0.5 at step 35,000, resumed on the October 5 original-camera
202-demo dataset (`FoxNerdSaysMoo/panthera-gear-carrier-pin-oct05-original-camera-20hz-202`).
This training stage uses a local dataset with no Hub revision; its manifest SHA-256
is `041690aea82ae8658c870159d12ba33717b2312a5683c3237f659764c6651b54`.

The enabled user service `panthera-inference` binds only `127.0.0.1:8765`.
User lingering keeps it available without a login. No AWS inbound port needs
opening beyond SSH. The service waits for `train_panthera.py` to exit before
allocating GPU memory. It then resolves `checkpoints/last/pretrained_model` once
and uses that immutable checkpoint until restarted. A failed load is reported
through `/health`; inspect the journal and restart after correcting the cause.

Concurrent inference is supported with `server.py --allow-training` after
checking free VRAM. The current AWS service has a `concurrent.conf` user-service
drop-in enabling this flag and selecting a preserved hard-linked snapshot of
step 35,000 at `/mnt/pi05/inference/checkpoints/panthera_pi05_20261003_035000`.
This snapshot survives the trainer's checkpoint pruning. Restarting the service
keeps that selected checkpoint; it does not pick up later training saves.
The prior step-26,250 model remains preserved at
`/mnt/pi05/inference/checkpoints/panthera_pi05_20261003_026250/pretrained_model`,
with its original deployment metadata and processors. Its inference contract is
saved alongside it as `inference-contract.json`. AWS rollback configuration is
in `/mnt/pi05/inference/backups/pre-035000/` (`contract.json` and `concurrent.conf`).
The older step-23,500 snapshot and `/mnt/pi05/inference/backups/pre-026250/`
rollback files also remain available.
Rolling back requires restoring both AWS files and the local contract before
restarting inference. Existing local clients load the contract at startup;
relaunch them safely and reconnect after a model change to refresh the identity.
Remove the remote drop-in and run `systemctl --user daemon-reload` followed by
`systemctl --user restart panthera-inference` to restore the default wait behavior.
Inference loads saved weights strictly, including omitted tied aliases, and
disables training-only gradient checkpointing.

From the repository root:

```bash
# Reinstall/update the AWS service and files:
./scripts/deploy-inference.sh
# Local robot environment has the existing SDK, numpy, yaml and OpenCV:
uv pip install --python work/teleop-venv/bin/python -r requirements-inference.txt
# Install the persistent local SSH tunnel (already done on this workstation):
./scripts/install-inference-tunnel.sh
# Alternatively, keep a foreground tunnel running in a terminal:
./scripts/inference-tunnel.sh
# Health and synthetic-image inference probe, without opening motor/camera devices:
work/teleop-venv/bin/python app/remote_inference.py
```

The probe exits 2 while training/loading is in progress. Successful inference
prints latency, checkpoint identity and the action count. Synthetic images test
the pipeline; they do not measure task performance. Transport-only mock servers
are explicitly rejected by physical execution.

Once `/health` is `ready`, stop hand guidance/Quest control, check the physical
workspace, keep the hardware stop accessible, and start a bounded session:

```bash
work/teleop-venv/bin/python app/remote_inference.py --run --seconds 30
```

`--run` enables both real arms and all three mapped cameras. Ctrl-C, termination,
camera/feedback loss, faults, stale inference, rejected commands, invalid targets,
or a missed local deadline call the SDK stop on both arms. The shared
`logs/control.lock` excludes the existing teleoperation controllers. The controller
holds the measured starting pose while requesting inference; inference runs in
a worker so networking never blocks the 150 Hz local loop. Policy actions are
20 Hz waypoints. A continuous quintic path passes through them in order, capped at
1.0 rad/s for joints and 5.0 rad/s for grippers, with acceleration caps of
8.0 rad/s² for joints and 30.0 rad/s² for grippers. Emitted target increments
are capped at 0.5 rad as well as the per-axis speed times the control interval.
Adjacent waypoints moving in the same direction share
velocity; direction reversals, repeated targets and chunk endpoints stop smoothly.
The path timing expands beyond 50 ms per action when needed, preserving all
waypoints. The existing conservative return torque limits apply. The command-line
runner requests the next chunk near the end of the current path, using measured
request latency to choose its lead time (at most one second). It buffers one
validated response and starts it at the chunk boundary. If inference is slower
than that lead time, it holds the last target until a fresh response arrives.
Both in-flight requests and buffered chunks retain the two-second observation
age limit. This is bounded prefetch, without RTC action alignment or model changes.

`contract.json` defines the exact arm order, task/dataset identity, fixed training
crops, and local limits. The client checks the server contract, camera age,
motor feedback/faults, joint/gripper limits, emitted command increments and tracking deviation.
Large in-limit prediction jumps are retimed rather than rejected.

Inference motion limits are 1.0 rad/s and 8.0 rad/s² for arm joints, and
5.0 rad/s and 30.0 rad/s² for grippers. A standalone 2-rad gripper stroke
plans in 0.75 seconds (previously 4.688 seconds); mixed arm/gripper paths
can take longer. Position bounds, torque caps, tracking checks and fault
handling remain in effect. Collection-pose resets still use the slower
collection settings. Both the remote server and local controller load the
contract at startup; synchronize the remote contract and restart both to
activate changes. A connected local controller defers shutdown to preserve
powered hold, so safely end its robot session before restarting it.
The 0.15 rad jump bound applies to each emitted command increment. Malformed and
nonfinite predictions reject the entire chunk before motion. Range checks cover
the selected action prefix for single-step dashboard execution, and the entire
chunk for preview or Play (including targets used for overlap blending). A future waypoint outside the selected prefix
does not block an otherwise valid step.
The server and client clamp all finite gripper predictions to the configured
stops (0 to 2 rad for this deployment), including large overshoots; returned
`limit_corrections` preserve each changed action index, raw target and command
target. All joint violations remain rejected. Measured gripper feedback still
uses the separate 0.03 rad encoder tolerance.
Limit errors identify the action, arm/joint, target and allowed range. Dashboard
logs retain prediction responses before validation, including rejected chunks.
It never converts gripper radians into a normalized open/close fraction. Wrist
camera names remain USB port names because physical left/right assignment was
unverified in the dataset. Predictions and latencies append to
`logs/inference-actions.jsonl`.

These bounds do not establish collision avoidance or successful manipulation.
The physical controller has not been validated through a real robot rollout;
the learned next-measured-pose actions are proxies for motor commands. Initial
physical sessions need supervision. The SSH/local software watchdog cannot stop
motors after a host crash, SIGKILL, or a stuck SDK call; use the hardware stop.

## Browser step controls

Start the local dashboard with `./scripts/start-inference-dashboard.sh`, then
open `http://127.0.0.1:8091`. Starting the web server alone opens no robot devices.
Use **Connect & reset collection pose** to check the policy contract, claim the shared
control lock, check all cameras and arm feedback, and hold the measured pose.
Set **Actions per step** from 1 to 50. **Execute** captures a fresh observation,
predicts a chunk, blends only the first selected number of action waypoints into
one continuous motion, and holds afterward. Actual execution can take longer
than the nominal 20 Hz policy time. The step starts and ends at rest; intermediate
waypoints share velocity when their direction agrees. Each click replans;
paused predictions are not replayed. Selecting one action still ends in a hold.
**Play** repeats **Actions per step** (1–50), capturing a fresh observation at
each step boundary. **Pause & hold** cancels the current motion and prediction,
discards the unused plan, and keeps the SDK connections in position hold. Play
after Pause always starts with a fresh observation; it never replays the canceled
response. **Apply settings** updates the action count and blend length for the
next prediction, including during playback. Single **Execute** remains one step.

**Blend actions** (0–50, default 10) crossfades the unused tail of the preceding
prediction into the new prediction with smoothstep weights. After executing N
waypoints, old row N aligns with new row 0; blending is limited by the available
overlap and fully transitions to the new plan at its end. Zero disables overlap
blending. A 50-action step leaves no overlap. Expired old predictions are discarded
rather than blended, using the existing two-second observation-age limit.
Convex combinations preserve validated position limits; the existing quintic
trajectory then enforces speed/acceleration bounds and starts/ends each step at
rest. Both arm joints and absolute gripper targets use the same crossfade.

This adapts the replan/overlap distinction from
[Dream Machines, section 2.10](https://dream-machines.eu/blog/pi05-fine-tuning).
It is client-side blending, not model-side RTC guidance. Each boundary has a brief
hold while inference runs; because the robot is holding during that request,
no actions elapse to subtract as an inference-latency offset. Retimed execution
also makes wall-clock 20 Hz offsets inappropriate. Predictions, effective blended
actions, overlap lengths and step completions are logged in
`logs/inference-steps.jsonl`. Inference/validation faults and browser loss disable
Play and clear the retained overlap; heartbeat recovery alone never resumes it.

**Preview prediction** predicts without changing the held pose. Speed,
acceleration, emitted-increment, torque, tracking, joint-limit and camera/feedback
checks apply. Retiming preserves the selected waypoint path without stopping at
every intermediate action.

**Pause & hold**, **Stop motion & hold**, Escape, closing the page, or losing
the browser heartbeat for two seconds cancels motion while retaining powered
position hold. Cancellation remains independent of the inference request worker.
Playback never resumes automatically; explicitly press Play or Execute after
resolving the cause. Hardware faults require the supported-arm recovery below. The web server is loopback-only, validates Host
and Origin, and requires a session token for commands. Step predictions and
completion events are saved to `logs/inference-steps.jsonl`.

The dashboard currently runs as `panthera-policy-dashboard.service` in the
user's service manager. Stop it with
`systemctl --user stop panthera-policy-dashboard`. It is a transient service,
so use the start script to launch it again after a reboot.

For USB startup failures, run `python3 scripts/diagnose_robot_usb.py`. It reports
the saved arm mapping, actual hub/board locations, configuration state, requested
USB current, serial interfaces and recent kernel power errors. It opens no motor
devices. `sudo python3 scripts/diagnose_robot_usb.py --hub-status` additionally
reads the hub's live USB power status using GET_STATUS requests; it does not
reset, configure, or rebind any device. A detected board with an empty USB
configuration and kernel power rejection has failed before the serial driver
can bind. Correct hub power/cabling first; a changed host port can additionally
cause a saved-mapping mismatch. Preserve physical left/right identity when
correcting topology. Do not force a USB configuration to bypass the power check.

The verified October 4 setup uses the Genesys `05e3:0610` hub on controller
`0000:08:00.4`: left arm in port 1, wrist cameras in ports 2 and 3, right arm
in port 4. The overhead camera connects directly to controller `0000:06:00.3`.
This uses two computer ports and passes the collection camera preflight at
640×480 and 20 Hz alongside both arm connections. The old `14cd:8601` hub
rejected configuration on power-budget grounds and its overhead camera still
failed USB communication after an explicitly requested temporary override;
it is no longer part of the active setup. `configure_robot_hub.py` was a
targeted diagnostic workaround for that old hub, not a startup requirement.

```bash
ssh 3.83.15.102 'systemctl --user status panthera-inference --no-pager'
ssh 3.83.15.102 'journalctl --user -u panthera-inference -n 100 --no-pager'
ssh 3.83.15.102 'systemctl --user restart panthera-inference'
curl http://127.0.0.1:8765/health
systemctl --user status panthera-inference-tunnel --no-pager
systemctl --user stop panthera-inference-tunnel
```

Override the SSH destination with `PANTHERA_SSH_HOST`. To change the model run,
edit the remote service checkpoint argument and redeploy; checkpoint metadata
must still match this deployment contract.

Inference connection now resets to the collection neutral pose from `config/recording.json` (`[0, 0.25, 0.25, 0, 0, 0]` radians for both arms, grippers 60% open). The shared `NeutralReturn` trajectory and collection path-clearance check are used, with joint speed capped at 0.6 rad/s and gripper speed at 0.8 rad/s. The dashboard shows `resetting` and blocks prediction/steps until measured arrival. `Reset collection pose` repeats the reset between runs; browser loss, Stop, tracking faults, or a gripper arrival timeout cancel motion and retain position hold.

Policy-limit rejection, inference errors, Stop, browser closure, and heartbeat loss retain the SDK connection and powered hold. Rejected chunks execute no actions. The **Zero arms (collection pose)** button returns to the collection pose; resets occur on connection, not between policy steps. Hardware feedback/command failures block further movement and retain SDK objects with best-effort hold, but an actual USB/power failure or killed process can prevent holding. SIGINT/SIGTERM defer shutdown while a robot session exists. The user-service drop-in `hold.conf` sets `TimeoutStopSec=infinity` and `SendSIGKILL=no`; a service stop/restart while connected therefore waits instead of releasing the arms. Physically support the arms before deliberately ending the process or removing power.


### Recovering failed robot connections

The dashboard runs the SDK session in a separate process so a native SDK crash
or blocked serial call does not freeze the browser server. **Reconnect robot
connections** is available after a hardware fault or while holding. First
physically support both arms and confirm the prompt: recovery releases powered
hold, closes the old session (terminating it if cleanup hangs), and opens fresh
connections only after the old process exits. It checks the current AWS policy
contract and current USB mapping, then holds the freshly measured pose. Recovery
never resets to the collection pose or replays a pending action. Use **Zero arms
(collection pose)** separately after checking clearance. **Stop motion & hold**
cancels an in-progress recovery; if the old connection has been released, both
arms must stay supported until a new connection is explicitly established.

A failed hold command is displayed as **Hold command failed**, rather than
implying that retained SDK objects establish a powered hold. If recovery fails,
check board power and USB cabling and retry Reconnect after correcting the cause.
No automatic recovery moves the robot. Reconnecting also refreshes the checkpoint
identity after an AWS model update. The web process still defers ordinary shutdown
while a robot session is alive; support the arms before deliberately releasing it.
