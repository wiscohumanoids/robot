#!/usr/bin/env python3
"""Job runner for the GPU VM: one tmux session per job, logs + state under ~/hsim_runs/<job>/.

Usage (on the VM; from the Mac use tools/vm_jobs.sh, which ssh's here):
  vmjobs.py [--jobs FILE] list
  vmjobs.py [--jobs FILE] launch <job> [<job> ...]
  vmjobs.py [--jobs FILE] launch-all            # every enabled job + the GPU logger
  vmjobs.py [--jobs FILE] status                # one table: progress, GPU, CPU/RAM/disk
  vmjobs.py [--jobs FILE] logs <job> [-f] [-n N]
  vmjobs.py [--jobs FILE] stop <job>            # Ctrl-C to the job (mjlab saves on its own schedule), then terminate
  vmjobs.py [--jobs FILE] resume <job>          # relaunch; rl jobs get --agent.resume True --agent.load-run <latest>
  vmjobs.py gpulog-start | gpulog-stop          # nvidia-smi every 60 s -> <runs>/gpu.csv
Internal: vmjobs.py _run <job> [--resume]      # what runs inside tmux

Needs only the system python3 + PyYAML (present on the VM). Job definitions: infra/vm_jobs/jobs.yaml.
"""
import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_JOBS = HERE / "jobs.yaml"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SESSION = "hsim-job-{}"
GPULOG_SESSION = "hsim-gpulog"


# ----------------------------------------------------------------------------- config
def load(jobs_file):
    cfg = yaml.safe_load(Path(jobs_file).read_text())
    d = cfg.get("defaults", {})
    base = {k: os.path.expanduser(str(d.get(k, v))) for k, v in
            (("repo", "~/humanoid-sim-stack"), ("runs", "~/hsim_runs"), ("data", "~/hsim_data"), ("envs", "~/hsim_envs"))}
    return cfg, base


def subst(s, base, job_id=None, extra=None):
    if s is None:
        return None
    s = str(s)
    vals = dict(base)
    if job_id:
        vals["log_root"] = os.path.join(base["runs"], job_id)
    vals["PATH"] = os.environ.get("PATH", "")
    vals.update(extra or {})
    for k, v in vals.items():
        s = s.replace("{" + k + "}", v)
    return re.sub(r"(^|[\s:=\"'])~/", lambda m: m.group(1) + os.path.expanduser("~") + "/", s)


def job_cfg(cfg, base, job_id):
    jobs = cfg.get("jobs", {})
    if job_id not in jobs:
        sys.exit(f"unknown job '{job_id}'. Jobs: {', '.join(jobs)}")
    j = dict(jobs[job_id])
    env = {k: subst(v, base, job_id) for k, v in {**cfg.get("defaults", {}).get("env", {}), **(j.get("env") or {})}.items()}
    j["env"] = env
    j["workdir"] = subst(j.get("workdir", "{repo}"), base, job_id)
    j["command"] = subst(j["command"], base, job_id)
    j["log_root"] = os.path.join(base["runs"], job_id)
    j.setdefault("kind", "step")
    j.setdefault("depends_on", [])
    j.setdefault("auto_resume", 0)
    j.setdefault("enabled", True)
    return j


# ----------------------------------------------------------------------------- state
def state_path(base, job_id):
    return Path(base["runs"]) / job_id / "state.json"


def read_state(base, job_id):
    p = state_path(base, job_id)
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def write_state(base, job_id, **kw):
    p = state_path(base, job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    s = read_state(base, job_id)
    s.update(kw)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(p)
    return s


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tmux_has(name):
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def latest_run_dir(log_root):
    """Newest directory under log_root that contains model_*.pt (mjlab: <log_root>/<experiment>/<timestamp>/)."""
    best = None
    for p in Path(log_root).rglob("model_*.pt"):
        d = p.parent
        if best is None or d.stat().st_mtime > best.stat().st_mtime:
            best = d
    return best


def latest_checkpoint(run_dir):
    cks = sorted(Path(run_dir).glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    return cks[-1] if cks else None


# ----------------------------------------------------------------------------- _run (inside tmux)
_child = None
_stop = False


def _forward(signum, frame):
    global _stop
    _stop = True
    if _child and _child.poll() is None:
        try:
            os.killpg(_child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def cmd_run(cfg, base, job_id, resume):
    global _child
    j = job_cfg(cfg, base, job_id)
    Path(j["log_root"]).mkdir(parents=True, exist_ok=True)
    log = open(Path(j["log_root"]) / "stdout.log", "a", buffering=1)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _forward)
    write_state(base, job_id, id=job_id, status="waiting" if j["depends_on"] else "starting",
                wrapper_pid=os.getpid(), started=None, ended=None, exit_code=None, stop_requested=False,
                launched=now(), attempts=read_state(base, job_id).get("attempts", 0) if resume else 0)

    # dependencies
    while j["depends_on"]:
        sts = {d: read_state(base, d).get("status") for d in j["depends_on"]}
        if all(v == "done" for v in sts.values()):
            break
        bad = {d: v for d, v in sts.items() if v in ("failed", "stopped", "blocked")}
        if bad or _stop:
            write_state(base, job_id, status="blocked" if bad else "stopped", ended=now(),
                        note=f"dependency not successful: {bad}" if bad else "stopped while waiting")
            print(f"[vmjobs] {job_id} blocked: {bad}", file=log)
            return 1
        time.sleep(30)

    attempt_resume = resume
    while True:
        cmd = j["command"]
        if attempt_resume and j["kind"] == "rl":
            rd = latest_run_dir(j["log_root"])
            if rd is not None:
                cmd += f" --agent.resume True --agent.load-run {rd.name}"
            else:
                print(f"[vmjobs] {job_id}: no checkpoint found to resume from; starting fresh", file=log)
        st = read_state(base, job_id)
        attempts = st.get("attempts", 0) + 1
        env = dict(os.environ)
        env.update(j["env"])
        header = f"\n[vmjobs] {now()} start {job_id} attempt {attempts}{' (resume)' if attempt_resume else ''}\n[vmjobs] cwd {j['workdir']}\n[vmjobs] cmd {cmd}\n"
        log.write(header)
        sys.stdout.write(header)
        _child = subprocess.Popen(["bash", "-c", cmd], cwd=j["workdir"], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, start_new_session=True, bufsize=1, text=True,
                                  errors="replace")
        write_state(base, job_id, status="running", started=now(), child_pid=_child.pid, attempts=attempts,
                    command=cmd, resumed=bool(attempt_resume))
        for line in _child.stdout:
            log.write(line)
            sys.stdout.write(line)
        code = _child.wait()
        stop_req = _stop or read_state(base, job_id).get("stop_requested", False)
        if stop_req:
            write_state(base, job_id, status="stopped", ended=now(), exit_code=code)
        elif code == 0:
            write_state(base, job_id, status="done", ended=now(), exit_code=0)
        else:
            write_state(base, job_id, status="failed", ended=now(), exit_code=code)
        msg = f"[vmjobs] {now()} {job_id} exited with code {code}{' (stop requested)' if stop_req else ''}\n"
        log.write(msg)
        sys.stdout.write(msg)
        if not stop_req and code != 0 and attempts <= j["auto_resume"]:
            log.write(f"[vmjobs] auto-resume ({attempts}/{j['auto_resume']}) in 30 s\n")
            time.sleep(30)
            attempt_resume = True
            continue
        return code


# ----------------------------------------------------------------------------- commands
def cmd_launch(cfg, base, jobs_file, job_ids, resume=False):
    for job_id in job_ids:
        j = job_cfg(cfg, base, job_id)
        name = SESSION.format(job_id)
        if tmux_has(name):
            print(f"{job_id}: already running (tmux {name}); use stop first")
            continue
        m = re.match(r"\s*(?:bash|sh|python3?)\s+(\S+)", j["command"])
        if m and not m.group(1).startswith("-"):
            script = Path(m.group(1)) if os.path.isabs(m.group(1)) else Path(j["workdir"]) / m.group(1)
            if not script.exists():
                print(f"{job_id}: NOT launched: {script} does not exist yet (owner still writing it? sync it first)")
                continue
        if not Path(j["workdir"]).is_dir():
            print(f"{job_id}: NOT launched: workdir {j['workdir']} does not exist")
            continue
        Path(j["log_root"]).mkdir(parents=True, exist_ok=True)
        if not resume:
            st = read_state(base, job_id)
            if st.get("status") in ("done",) and latest_run_dir(j["log_root"]) is not None:
                print(f"{job_id}: note: previous run finished; launching a fresh run (use `resume` to continue it)")
        args = [sys.executable, str(Path(__file__).resolve()), "--jobs", str(Path(jobs_file).resolve()), "_run", job_id]
        if resume:
            args.append("--resume")
        inner = " ".join(f"'{a}'" for a in args)
        subprocess.run(["tmux", "new-session", "-d", "-s", name, "-x", "220", "-y", "50", inner], check=True)
        print(f"{job_id}: launched in tmux session {name}{' (resume)' if resume else ''}; log {j['log_root']}/stdout.log")


def cmd_stop(cfg, base, job_id, timeout=45):
    name = SESSION.format(job_id)
    st = write_state(base, job_id, stop_requested=True)
    pid = st.get("child_pid")
    if pid:
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    wpid = st.get("wrapper_pid")
    t0 = time.time()
    while tmux_has(name) and time.time() - t0 < timeout:
        time.sleep(1)
    if tmux_has(name):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in (pid,):
                if p:
                    try:
                        os.killpg(p, sig)
                    except ProcessLookupError:
                        pass
            time.sleep(5)
            if not tmux_has(name):
                break
        subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
        if read_state(base, job_id).get("status") in ("running", "waiting", "starting"):
            write_state(base, job_id, status="stopped", ended=now())
    print(f"{job_id}: stopped (status {read_state(base, job_id).get('status')})")


def tail_text(path, nbytes=262144):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return ANSI.sub("", f.read().decode("utf-8", "replace"))
    except FileNotFoundError:
        return ""


def parse_rl(text):
    i = text.rfind("Learning iteration")
    if i < 0:
        return None
    blk = text[i:]
    m = re.search(r"Learning iteration (\d+)/(\d+)", blk)
    out = {"it": int(m.group(1)), "total": int(m.group(2))}
    for key, pat in (("reward", r"Mean reward:\s*(-?[\d.]+)"), ("eplen", r"Mean episode length:\s*([\d.]+)"),
                     ("sps", r"Steps per second:\s*([\d.]+)"), ("eta", r"ETA:\s*([0-9:, a-z]+)"),
                     ("itime", r"Iteration time:\s*([\d.]+)s")):
        mm = re.search(pat, blk)
        out[key] = mm.group(1).strip() if mm else ""
    return out


def proc_parents():
    par = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
                par[int(p.name)] = int(fields[1])
            except Exception:
                pass
    return par


def gpu_by_job(states):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}, []
    par = proc_parents()
    roots = {}
    for jid, st in states.items():
        for k in ("child_pid", "wrapper_pid"):
            if st.get(k):
                roots[int(st[k])] = jid
    used, other = {}, []
    for line in out.strip().splitlines():
        try:
            pid, mem = [int(x.strip()) for x in line.split(",")]
        except ValueError:
            continue
        p, jid = pid, None
        for _ in range(50):
            if p in roots:
                jid = roots[p]
                break
            p = par.get(p, 0)
            if p <= 1:
                break
        if jid:
            used[jid] = used.get(jid, 0) + mem
        else:
            other.append((pid, mem))
    return used, other


def system_line(base):
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        util, mu, mt, temp = [x.strip() for x in g.split(",")]
        gpu = f"GPU {util}% util, {int(mu) / 1024:.1f}/{int(mt) / 1024:.0f} GB, {temp} C"
    except Exception:
        gpu = "GPU n/a"
    load = open("/proc/loadavg").read().split()[:3]
    mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable")}
    st = os.statvfs(base["runs"] if os.path.isdir(base["runs"]) else os.path.expanduser("~"))
    disk_free = st.f_bavail * st.f_frsize / 1e9
    return (f"{gpu} | CPU load {' '.join(load)} ({os.cpu_count()} cores) | RAM {(mi['MemTotal'] - mi['MemAvailable']) / 1e6:.1f}/"
            f"{mi['MemTotal'] / 1e6:.0f} GB | disk free {disk_free:.0f} GB")


def cmd_status(cfg, base):
    jobs = cfg.get("jobs", {})
    states = {jid: read_state(base, jid) for jid in jobs}
    gpu, other = gpu_by_job(states)
    rows = [("job", "state", "iter", "reward", "ep_len", "steps/s", "ETA", "GPU GB", "info")]
    for jid, jraw in jobs.items():
        st = states[jid]
        status = st.get("status", "-" if jraw.get("enabled", True) else "disabled")
        if status in ("running", "waiting", "starting") and not tmux_has(SESSION.format(jid)):
            status += "?"   # session gone without a final state: crashed wrapper
        text = tail_text(Path(base["runs"]) / jid / "stdout.log")
        rl = parse_rl(text) if jraw.get("kind") == "rl" else None
        g = f"{gpu[jid] / 1024:.1f}" if jid in gpu else ""
        if rl:
            rows.append((jid, status, f"{rl['it']}/{rl['total']}", rl["reward"], rl["eplen"], rl["sps"], rl["eta"], g,
                         f"attempt {st.get('attempts', '')}" + (f", exit {st['exit_code']}" if st.get("exit_code") not in (None, 0) else "")))
        else:
            last = next((l.strip() for l in reversed(text.splitlines()) if l.strip()), "")
            info = (st.get("note") or last)[:70]
            if st.get("exit_code") not in (None, 0):
                info = f"exit {st['exit_code']}: " + info
            rows.append((jid, status, "", "", "", "", "", g, info))
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]) - 1)]
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "|", system_line(base))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r[:-1])) + "  " + str(r[-1]))
    if other:
        print("other GPU processes (not hsim jobs): " + ", ".join(f"pid {p} {m / 1024:.1f} GB" for p, m in other))
    print(f"GPU logger: {'running' if tmux_has(GPULOG_SESSION) else 'off'} ({base['runs']}/gpu.csv)")


def cmd_logs(base, job_id, follow, n):
    p = Path(base["runs"]) / job_id / "stdout.log"
    if not p.exists():
        sys.exit(f"no log yet: {p}")
    os.execvp("tail", ["tail", "-n", str(n)] + (["-F"] if follow else []) + [str(p)])


def cmd_gpulog(base, start):
    if not start:
        subprocess.run(["tmux", "kill-session", "-t", GPULOG_SESSION], capture_output=True)
        print("GPU logger stopped")
        return
    if tmux_has(GPULOG_SESSION):
        print("GPU logger already running")
        return
    Path(base["runs"]).mkdir(parents=True, exist_ok=True)
    f = os.path.join(base["runs"], "gpu.csv")
    sh = (f"[ -s {f} ] || echo 'timestamp, util_pct, mem_used_mib, mem_total_mib, temp_c, power_w' > {f}; "
          f"nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
          f"--format=csv,noheader,nounits -l 60 >> {f}")
    subprocess.run(["tmux", "new-session", "-d", "-s", GPULOG_SESSION, sh], check=True)
    print(f"GPU logger started -> {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default=os.environ.get("HSIM_JOBS", str(DEFAULT_JOBS)))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("launch"); p.add_argument("job", nargs="+")
    sub.add_parser("launch-all")
    sub.add_parser("status")
    p = sub.add_parser("logs"); p.add_argument("job"); p.add_argument("-f", action="store_true"); p.add_argument("-n", type=int, default=60)
    p = sub.add_parser("stop"); p.add_argument("job", nargs="+")
    p = sub.add_parser("resume"); p.add_argument("job", nargs="+")
    sub.add_parser("gpulog-start"); sub.add_parser("gpulog-stop")
    p = sub.add_parser("_run"); p.add_argument("job"); p.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg, base = load(a.jobs)
    if a.cmd == "list":
        for jid, j in cfg.get("jobs", {}).items():
            print(f"{jid:26s} {'enabled ' if j.get('enabled', True) else 'disabled'} {j.get('kind', 'step'):7s} "
                  f"gpu~{j.get('gpu_mem_gb', '?')} GB  deps={j.get('depends_on', [])}  {j.get('description', '')}")
    elif a.cmd == "launch":
        cmd_launch(cfg, base, a.jobs, a.job)
    elif a.cmd == "launch-all":
        cmd_gpulog(base, True)
        enabled = [jid for jid, j in cfg.get("jobs", {}).items() if j.get("enabled", True)]
        cmd_launch(cfg, base, a.jobs, enabled)
    elif a.cmd == "status":
        cmd_status(cfg, base)
    elif a.cmd == "logs":
        cmd_logs(base, a.job, a.f, a.n)
    elif a.cmd == "stop":
        for jid in a.job:
            cmd_stop(cfg, base, jid)
    elif a.cmd == "resume":
        cmd_launch(cfg, base, a.jobs, a.job, resume=True)
    elif a.cmd == "gpulog-start":
        cmd_gpulog(base, True)
    elif a.cmd == "gpulog-stop":
        cmd_gpulog(base, False)
    elif a.cmd == "_run":
        sys.exit(cmd_run(cfg, base, a.job, a.resume))


if __name__ == "__main__":
    main()
