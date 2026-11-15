"""Morning report for the overnight jobs: <runs>/report/index.md + index.html.

For every rl job in jobs.yaml that has a checkpoint:
  - reward + episode-length curves (png) from its stdout.log (falls back to TensorBoard events)
  - the ONNX mjlab exported next to the latest checkpoint (copied as policy.onnx)
  - a short headless video (record_video.py, EGL) of the latest checkpoint
  - the job's eval command, if `report.eval` is set ({checkpoint} is replaced), output in eval.txt
Other jobs: state + last log lines.

Run on the VM with the mjlab venv (needs matplotlib, mjlab):
  cd ~/humanoid-sim-stack && MUJOCO_GL=egl training/locomotion/.venv/bin/python infra/vm_jobs/morning_report.py
      [--jobs infra/vm_jobs/jobs.yaml] [--only J1,J3] [--no-video] [--steps 500]
"""
import argparse
import datetime as dt
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vmjobs  # noqa: E402  (config + state helpers)


def curves_from_stdout(log):
    text = vmjobs.tail_text(log, nbytes=1 << 31)
    its, rew, eplen = [], [], []
    for blk in text.split("Learning iteration ")[1:]:
        m = re.match(r"(\d+)/", blk)
        r = re.search(r"Mean reward:\s*(-?[\d.]+)", blk)
        e = re.search(r"Mean episode length:\s*([\d.]+)", blk)
        if m and r:
            its.append(int(m.group(1))); rew.append(float(r.group(1))); eplen.append(float(e.group(1)) if e else float("nan"))
    return its, rew, eplen


def curves_from_tb(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return [], [], []
    ea = EventAccumulator(str(run_dir)); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "Train/mean_reward" not in tags:
        return [], [], []
    r = ea.Scalars("Train/mean_reward")
    e = {s.step: s.value for s in ea.Scalars("Train/mean_episode_length")} if "Train/mean_episode_length" in tags else {}
    return [s.step for s in r], [s.value for s in r], [e.get(s.step, float("nan")) for s in r]


def plot(its, rew, eplen, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(its, rew, lw=1); ax[0].set_title("mean reward"); ax[0].set_xlabel("iteration")
    ax[1].plot(its, eplen, lw=1, color="tab:orange"); ax[1].set_title("mean episode length"); ax[1].set_xlabel("iteration")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(vmjobs.DEFAULT_JOBS))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    cfg, base = vmjobs.load(a.jobs)
    only = set(x for x in a.only.split(",") if x)
    rep_dir = Path(base["runs"]) / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# Overnight report, generated {dt.datetime.now():%Y-%m-%d %H:%M}", "",
          "| job | status | iterations | final reward | final ep. length | checkpoint | video | eval |", "|---|---|---|---|---|---|---|---|"]
    sections = []
    for jid in cfg.get("jobs", {}):
        if only and jid not in only:
            continue
        j = vmjobs.job_cfg(cfg, base, jid)
        st = vmjobs.read_state(base, jid)
        if not st and not Path(j["log_root"]).exists():
            continue
        out = rep_dir / jid
        out.mkdir(exist_ok=True)
        sec = [f"## {jid}", "", f"{j.get('description', '')}", "", f"status: **{st.get('status', 'not run')}**, "
               f"attempts {st.get('attempts', 0)}, started {st.get('started')}, ended {st.get('ended')}", ""]
        if j["kind"] != "rl":
            last = vmjobs.tail_text(Path(j["log_root"]) / "stdout.log").strip().splitlines()[-15:]
            sec += ["```", *last, "```", ""]
            md.append(f"| {jid} | {st.get('status', '-')} | | | | | | |")
            sections.append("\n".join(sec)); continue
        run_dir = vmjobs.latest_run_dir(j["log_root"])
        ck = vmjobs.latest_checkpoint(run_dir) if run_dir else None
        its, rew, eplen = curves_from_stdout(Path(j["log_root"]) / "stdout.log")
        if not its and run_dir:
            its, rew, eplen = curves_from_tb(run_dir)
        if its:
            plot(its, rew, eplen, out / "curves.png", jid)
            sec += ["![curves](" + f"{jid}/curves.png)", ""]
        onnx = sorted(run_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime) if run_dir else []
        if onnx:
            shutil.copy2(onnx[-1], out / "policy.onnx")
            sec.append(f"ONNX: `{jid}/policy.onnx` (from {onnx[-1].name}, exported by mjlab at the last save)")
        if ck:
            (out / "checkpoint.txt").write_text(str(ck) + "\n")
            sec.append(f"latest checkpoint: `{ck}`")
        rcfg = {k: vmjobs.subst(v, base, jid) for k, v in (j.get("report") or {}).items()}
        venv_py = os.path.join(rcfg.get("venv", f"{base['repo']}/training/locomotion/.venv"), "bin", "python")
        video, ev = "", ""
        if ck and not a.no_video and rcfg.get("task"):
            cmd = [venv_py, str(HERE / "record_video.py"), "--task", rcfg["task"], "--checkpoint", str(ck),
                   "--out", str(out), "--steps", str(a.steps)]
            if rcfg.get("import_path"):
                cmd += ["--import-path", rcfg["import_path"]]
            if rcfg.get("import_module"):
                cmd += ["--import-module", rcfg["import_module"]]
            if rcfg.get("motion_file"):
                cmd += ["--motion-file", rcfg["motion_file"]]
            r = subprocess.run(cmd, cwd=j["workdir"], capture_output=True, text=True, timeout=1800,
                               env={**os.environ, "MUJOCO_GL": "egl"})
            (out / "video_log.txt").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
            video = "yes" if (out / "video.mp4").exists() else "FAILED (video_log.txt)"
            if (out / "video.mp4").exists():
                sec.append(f"video: `{jid}/video.mp4`")
        if ck and rcfg.get("eval"):
            ecmd = rcfg["eval"].replace("{checkpoint}", str(ck))
            r = subprocess.run(["bash", "-c", ecmd], cwd=j["workdir"], capture_output=True, text=True, timeout=1800,
                               env={**os.environ, **j["env"]})
            (out / "eval.txt").write_text(r.stdout + "\n" + r.stderr[-5000:])
            ev = "ok" if r.returncode == 0 else f"exit {r.returncode}"
            sec += ["", "eval:", "```", *r.stdout.strip().splitlines()[-30:], "```"]
        md.append(f"| {jid} | {st.get('status', '-')} | {its[-1] if its else ''} | {f'{rew[-1]:.2f}' if rew else ''} | "
                  f"{f'{eplen[-1]:.1f}' if eplen else ''} | {ck.name if ck else ''} | {video} | {ev} |")
        sections.append("\n".join(sec))
    gpu = Path(base["runs"]) / "gpu.csv"
    if gpu.exists():
        md += ["", f"GPU log: `{gpu}`"]
    text = "\n".join(md) + "\n\n" + "\n\n".join(sections) + "\n"
    (rep_dir / "index.md").write_text(text)
    # minimal html: same content, images and videos inline
    body = []
    for line in text.splitlines():
        m = re.match(r"!\[curves\]\((.+)\)", line)
        if m:
            body.append(f'<img src="{m.group(1)}" style="max-width:100%">')
        elif line.startswith("video: `"):
            p = line.split("`")[1]
            body.append(f'<video src="{p}" controls style="max-width:640px"></video>')
        elif line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            body.append(f"<h{n}>{html.escape(line.lstrip('#').strip())}</h{n}>")
        else:
            body.append(f"<div>{html.escape(line)}</div>" if line.strip() else "<br>")
    (rep_dir / "index.html").write_text("<!doctype html><meta charset=utf-8><title>Overnight report</title>"
                                        "<body style='font-family:sans-serif;max-width:1000px;margin:auto'>"
                                        + "\n".join(body) + "</body>")
    print(f"wrote {rep_dir / 'index.md'} and index.html")


if __name__ == "__main__":
    main()
