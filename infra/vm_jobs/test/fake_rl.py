"""Dummy RL job for testing vmjobs: prints rsl_rl-style blocks and writes fake checkpoints.
Args: --log-root DIR [--iters N] [--fail-at K] [--agent.resume True --agent.load-run NAME]"""
import argparse, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--log-root"); ap.add_argument("--iters", type=int, default=40); ap.add_argument("--fail-at", type=int, default=-1)
ap.add_argument("--agent.resume", dest="resume", default="False"); ap.add_argument("--agent.load-run", dest="load_run", default=None)
a = ap.parse_args()
start = 0
run = os.path.join(a.log_root, "fake_exp", "run_resume" if a.resume == "True" else "run_0")
if a.resume == "True":
    prev = os.path.join(a.log_root, "fake_exp", a.load_run)
    ck = sorted(int(f[6:-3]) for f in os.listdir(prev) if f.startswith("model_"))
    start = ck[-1] + 1
    print(f"[INFO] resuming from {prev} model_{ck[-1]}.pt", flush=True)
os.makedirs(run, exist_ok=True)
for it in range(start, a.iters):
    print("#" * 80 + f"\n\033[1m Learning iteration {it}/{a.iters} \033[0m\n\n   Steps per second: 51234 \n"
          f"   Mean reward: {it * 0.5:.2f}\n   Mean episode length: {100 + it:.2f}\n" + "-" * 80 +
          f"\n   Iteration time: 0.20s\n   ETA: 0:00:{a.iters - it:02d}\n", flush=True)
    if it % 5 == 0:
        open(os.path.join(run, f"model_{it}.pt"), "w").write("x")
    if it == a.fail_at and a.resume != "True":
        print("simulated crash", flush=True); sys.exit(3)
    time.sleep(0.2)
print("done", flush=True)
