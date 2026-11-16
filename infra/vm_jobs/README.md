# VM jobs: run, watch, pull, report

Overnight job control for the GPU VM (`hsim-gpu`). Jobs are defined in `jobs.yaml`; each runs in its own
tmux session `hsim-job-<id>` so it survives ssh disconnects. Everything a job produces goes to
`~/hsim_runs/<id>/` on the VM (persistent EBS disk).

| File | Where it runs | What |
|---|---|---|
| `jobs.yaml` | | Job definitions: workdir, command, env, GPU estimate, depends_on, auto_resume, report settings |
| `vmjobs.py` | VM (system python3 + PyYAML) | The runner: launch / status / logs / stop / resume, GPU logger |
| `../../tools/vm_jobs.sh` | Mac or VM | Same commands everywhere; on the Mac it ssh's to the VM |
| `record_video.py` | VM (mjlab venv) | Headless video of a trained policy (mjlab's `play --video` opens a viewer and never exits headless) |
| `morning_report.py` | VM (mjlab venv) | Curves, ONNX, video, eval per RL job -> `~/hsim_runs/report/index.md` + `index.html` |
| `pull_manifest.sh` + `../../tools/vm_pull_results.sh` | VM + Mac | Pull only small results to the Mac (`runs/2026-11-10/`) |
| `test/` | VM | Dummy jobs used to test all of the above (`fake_rl.py` prints rsl_rl-style output) |

## Commands (from the Mac)
```
tools/vm_jobs.sh list                       # jobs, enabled?, kind, GPU estimate, dependencies
tools/vm_jobs.sh launch J1_g1_vel_arms_free        # one or more jobs
tools/vm_jobs.sh launch-all                 # every enabled job + the GPU logger (~/hsim_runs/gpu.csv, every 60 s)
tools/vm_jobs.sh status                     # table: iter, reward, ep length, steps/s, ETA, GPU GB per job + GPU/CPU/RAM/disk
tools/vm_jobs.sh logs J3_wisco_v1_vel -f    # follow a job's stdout (-n N for the last N lines)
tools/vm_jobs.sh stop J2b_g1_track_jumps1   # Ctrl-C to the job's process group, then terminate after 45 s
tools/vm_jobs.sh resume J1_g1_vel_arms_free        # rl jobs: adds --agent.resume True --agent.load-run <newest run dir>
ssh hsim-gpu 'cd ~/humanoid-sim-stack && MUJOCO_GL=egl training/locomotion/.venv/bin/python infra/vm_jobs/morning_report.py'
tools/vm_pull_results.sh                    # -> runs/2026-11-10/ (use --list to see what would come)
```

## Behaviour
- **States** (`~/hsim_runs/<id>/state.json`): waiting (for dependencies), running, done (exit 0), failed, stopped, blocked
  (a dependency failed or was stopped). `status` shows `running?` if the tmux session vanished without a final state.
- **depends_on**: the job waits (checks every 30 s) until each dependency is `done`; used for the manipulation chain
  M1_datagen -> M1_convert -> M1_finetune.
- **auto_resume: n**: after a crash (nonzero exit that wasn't a `stop`), relaunch with resume args up to n times, 30 s later.
  mjlab saves a checkpoint every 50 iterations, so a crash loses at most 50 iterations.
- **launch refuses** if the job's script (`bash <script>`) or workdir doesn't exist yet, or if the job is already running.
- **Placeholders** in commands/env: `{repo}` `{runs}` `{log_root}` (= `{runs}/<id>`) `{data}` `{envs}`; env `PATH` gets
  `~/.local/bin` (uv) prepended; `HF_HOME=~/hsim_data/hf-cache`; `MUJOCO_GL=egl`; `WANDB_MODE=offline`.
- **Status parsing** reads the last 256 KB of `stdout.log` (rsl_rl's per-iteration block: `Learning iteration i/N`,
  `Mean reward`, `Mean episode length`, `Steps per second`, `ETA`). GPU memory is credited to a job by walking each
  GPU process's parent chain up to the job's process.
- **Morning report**: per RL job, curves from stdout.log (falls back to TensorBoard events), the newest ONNX mjlab
  exported next to its checkpoints (mjlab's velocity and tracking runners export ONNX at every save), a 500-step video,
  and `report.eval` (with `{checkpoint}`) if set. Tested on a copy of the 2026-11-08 G1 all-joints run: 53 s per job incl. video + eval.
- **Pull**: the VM lists `*.onnx *.json *.yaml *.md *.html *.png *.mp4 *.csv *.txt` + the newest `model_*.pt` per run dir,
  skipping `data/ datasets/ raw/ wandb/ vm_backup/`, stdout logs and TensorBoard events; the Mac checks free space
  (keeps 5 GB spare) and streams the files over ssh with tar (macOS ships openrsync, which lacks needed rsync features).

## Filling in the placeholders (owners)
- B: confirm `J1` task id / module and its eval command. C: confirm clip file names and motion-file paths (`J2a-c`, `J4`),
  the Wisco tracking task id/module. A: confirm `J3` task id/module. E: `S1_perception` command, set `enabled: true`.
  F1/F2: `M1_*` commands, set `enabled: true`.

## Re-test
```
J=/home/ubuntu/humanoid-sim-stack/infra/vm_jobs/test/jobs_test.yaml
tools/vm_jobs.sh --jobs $J launch T2_after_T1 T1_rl T3_crash_resume T4_long   # deps, crash+auto-resume, stop target
tools/vm_jobs.sh --jobs $J status; tools/vm_jobs.sh --jobs $J stop T4_long
ssh hsim-gpu 'rm -rf ~/hsim_runs/_vmjobs_test'
```
Never test the guard or anything else with the real `jobs.yaml`: a job whose script exists launches for real.
