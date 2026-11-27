# vm_jobs — the GPU machine's job queue

Jobs (training runs, servers, pipeline steps) are defined in `jobs.yaml`; each runs in its own tmux session
`hsim-job-<id>`, so it survives ssh disconnects, and writes everything to `~/hsim_runs/<id>/` on the machine.

| File | Runs on | What |
|---|---|---|
| `jobs.yaml` | | job definitions: workdir, command, env, GPU estimate, dependencies, auto-resume, report settings (format at the top of the file) |
| `vmjobs.py` | GPU machine (system python3 + PyYAML) | the runner: launch, status, logs, stop, resume, GPU logger |
| `record_video.py` | GPU machine (mjlab venv) | headless video of a trained policy |
| `eval_velocity.py` | GPU machine (mjlab venv) | scripted command tests of a walking checkpoint |
| `morning_report.py` | GPU machine (mjlab venv) | per training job: curves, the newest ONNX, a video and evaluation → `~/hsim_runs/report/index.md` + `index.html` |
| `pull_manifest.sh` | GPU machine | lists the small results worth copying back (`tools/dev/vm.sh pull-results`) |
| `test/` | GPU machine | dummy jobs that exercise the runner (`fake_rl.py` prints training-style output) |

## Commands (from the laptop)

```
tools/dev/vm.sh jobs list                    # jobs, enabled, kind, GPU estimate, dependencies
tools/dev/vm.sh jobs launch wisco_velocity   # one or more jobs
tools/dev/vm.sh jobs launch-all              # every enabled job + the GPU logger (~/hsim_runs/gpu.csv every 60 s)
tools/dev/vm.sh jobs status                  # iteration, reward, episode length, steps/s, ETA, GPU memory per job
tools/dev/vm.sh jobs logs wisco_velocity -f  # follow a job's output (-n N for the last N lines)
tools/dev/vm.sh jobs stop wisco_velocity     # Ctrl-C to the job's process group, terminate after 45 s
tools/dev/vm.sh jobs resume wisco_velocity   # training jobs: --agent.resume True --agent.load-run <newest run>
ssh hsim-gpu 'cd ~/robot && MUJOCO_GL=egl training/locomotion/.venv/bin/python infra/vm_jobs/morning_report.py'
tools/dev/vm.sh pull-results                 # copy small results to results/ (--list to preview)
```
On the GPU machine itself, run the runner directly: `python3 ~/robot/infra/vm_jobs/vmjobs.py <command>`.

## Behavior

- **States** (`~/hsim_runs/<id>/state.json`): waiting (for dependencies), running, done (exit 0), failed, stopped,
  blocked (a dependency failed or was stopped). `status` shows `running?` if the tmux session vanished without a final state.
- **depends_on**: the job waits (checking every 30 s) until each dependency is done, e.g. data generation → conversion →
  fine-tuning.
- **auto_resume: n**: after a crash (nonzero exit that wasn't a `stop`), relaunch with resume arguments up to n times.
- **launch refuses** if the job's script or workdir doesn't exist, or the job is already running.
- **Status parsing** reads the end of `stdout.log` (RSL-RL's per-iteration block: iteration, mean reward, mean episode
  length, steps per second, ETA). GPU memory is credited to a job by walking each GPU process's parent chain.
- **Morning report**: curves from `stdout.log` (or TensorBoard events), the newest exported ONNX, a 500-step video, and
  the job's `report.eval` command if set.
- **Pull**: ONNX, JSON, YAML, Markdown, HTML, PNG, MP4, CSV and text files plus the newest checkpoint per run; never
  datasets, raw data, W&B folders, stdout logs or TensorBoard events. The laptop keeps 5 GB free.

## Testing the runner

```
tools/dev/vm.sh jobs --jobs ~/robot/infra/vm_jobs/test/jobs_test.yaml launch T2_after_T1 T1_rl T3_crash_resume T4_long
tools/dev/vm.sh jobs --jobs ~/robot/infra/vm_jobs/test/jobs_test.yaml status
ssh hsim-gpu 'rm -rf ~/hsim_runs/_vmjobs_test'
```
Never test with the real `jobs.yaml`: a job whose script exists launches for real.
