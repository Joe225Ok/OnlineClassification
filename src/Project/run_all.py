"""Run the whole project end to end as scripts: stage 0 (parsing) and 1 (tracks) once, stages 2 to 5 and the export of
the artifacts for each track, then the uncertainty (MC dropout) and drift (PSI) analyses for Track A.

The stages are the scripts in pipeline/ (s0_prepare.py ... s7_monitoring.py and export_artifacts.py); each step runs in
its own process with the project folder as working directory. Step 6 copies each trained model, with its
standardisation, temperatures, OOD detectors and thresholds, into artifacts/<track>/<model>/, which pipeline/infer.py
uses to classify new captures. Steps 7 (s6_uncertainty: MC dropout) and 8 (s7_monitoring: PSI) run for Track A only, on
the model without the testbed group.

    python run_all.py                  # every stage, Tracks A and B, one step at a time
    python run_all.py --jobs 2         # run independent steps in parallel (see below)
    python run_all.py --tracks A       # one track
    python run_all.py --from 3         # start at stage 3 (earlier results are read from disk)
    python run_all.py --only 2 3       # just these stages
    python run_all.py --quick DIR      # smoke test of stages 2-5 on a tiny subsample, results in DIR
    python run_all.py --yes (or -y)    # unattended: answer yes to the overwrite question

Before the first step the run lists every existing file the selected steps rewrite (results tables, tracks, units,
artifacts) with the consequences and asks yes or no; trained models and parsed data are caches and are never
overwritten (see safeguard.py). On yes those files are copied to runs/backup/<time>/ first; if a step fails, its files
are put back from that copy. --quick writes only to DIR and asks nothing.

Order: 0 -> 1 -> 2 (per track) -> 3, 4 and 5 (per track, each needs only that track's stage 2) -> 6 (after 3) -> 7 and
8 (Track A, after 3). With --jobs N, up to N steps whose inputs are ready run at once (the two tracks, and stages 3-5 of
a track), and the CPU threads are split between them. Each job needs several GB of memory (stage 3 about 9 GB), so on
a 32 GB laptop use at most 2. Results are bit-identical between runs with the same --jobs; a different thread count can
change the last digits.

Each step runs as its own process, so memory is freed between steps; output goes to runs/logs/<step>.log and the
time each step took to runs/logs/run_all_times.csv. No new step starts after one fails. Before the first step, the
machine (CPU, cores, memory), the thread settings and the package versions are written to data/results/environment.txt
(see environment.py), with a dated copy in runs/logs/.
Models and results are cached: delete data/models_v2, data/results and data/tracks to recompute everything.
"""
import argparse
import csv
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"
sys.path.insert(0, str(ROOT))
import safeguard

STAGES = {0: ("s0_prepare", False), 1: ("s1_data", False), 2: ("s2_models", True), 3: ("s3_evaluate", True),
          4: ("s4_experiments", True), 5: ("s5_stateful", True), 6: ("export_artifacts", True),
          7: ("s6_uncertainty", True), 8: ("s7_monitoring", True)}  # script, runs once per track
NEEDS = {0: None, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}  # the stage each one reads from
TRACK_A_ONLY = {7, 8}  # MC dropout and PSI: Track A, model without the testbed group (ROLE=no_testbed)


def keep_awake():
    """Ask Windows not to sleep while this process runs (the request ends when it exits). No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except (AttributeError, OSError):
        pass  # the request is a convenience only; a run must not fail because it was refused


def tracks_of(stage, tracks):
    """The tracks a stage runs for: none for the shared stages, Track A only for the analyses of the chosen model."""
    if not STAGES[stage][1]:
        return [None]
    return [t for t in tracks if t == "A"] if stage in TRACK_A_ONLY else tracks


def plan(stages, tracks):
    """Steps as (name, stage, track), with the step each one waits for (if it is part of this run)."""
    steps = []
    for s in sorted(stages):
        for t in tracks_of(s, tracks):
            steps.append((STAGES[s][0] + (f"_{t}" if t else ""), s, t))
    names = {(s, t) for _, s, t in steps}
    wait = {}
    for name, s, t in steps:
        need = NEEDS[s]
        dep = None if need is None else (need, t if STAGES[need][1] else None)
        wait[name] = next((n for n, s2, t2 in steps if dep and (s2, t2) == dep), None) if dep in names else None
    return steps, wait


def record_environment(folder, logs, env, jobs, threads):
    """Write folder/environment.txt (machine, thread settings, packages; see environment.py) and keep a dated copy in
    the logs, since a partial run rewrites the file. Run as a child with the steps' environment, so the thread pools
    it reports are those the steps get. A failure here only warns: it must not stop the run."""
    target = folder / "environment.txt"
    cmd = [sys.executable, str(ROOT / "environment.py"), "--jobs", str(jobs), "--threads", threads,
           "--command", " ".join(["python", "run_all.py", *sys.argv[1:]]), "--out", str(target)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"warning: {target} not written ({(r.stderr or r.stdout).strip().splitlines()[-1:]})", flush=True)
        return
    copy = logs / f"environment_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    shutil.copyfile(target, copy)
    print(f"environment recorded in {target} (copy: {copy.name})", flush=True)


def command(stage):
    return [sys.executable, str(PIPELINE / f"{STAGES[stage][0]}.py")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--from", dest="start", type=int, default=0, help="first stage to run")
    ap.add_argument("--only", nargs="+", type=int, choices=list(STAGES), help="run only these stages")
    ap.add_argument("--jobs", type=int, default=1, help="steps run at once (default 1)")
    ap.add_argument("--quick", metavar="DIR", help="smoke test of stages 2-5: tiny subsample, one epoch, results in DIR")
    ap.add_argument("-y", "--yes", action="store_true", help="answer yes to the overwrite question (unattended runs)")
    args = ap.parse_args()
    keep_awake()

    out_dir = Path(args.quick).resolve() if args.quick else ROOT / "runs"
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    times = logs / "run_all_times.csv"
    threads = str(max(1, (os.cpu_count() or 8) // args.jobs))
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": threads,
           "MKL_NUM_THREADS": threads}
    if args.quick:
        env["QUICK"] = str(out_dir)
    stages = args.only or [s for s in STAGES if s >= args.start]
    if args.quick:
        stages = [s for s in stages if s >= 2]
    steps, wait = plan(stages, args.tracks)
    globs = {name: safeguard.rewrites(STAGES[s][0], t) for name, s, t in steps}
    backup = None
    if not args.quick:
        if not safeguard.confirm(safeguard.describe([(n, globs[n]) for n, _, _ in steps], ROOT), args.yes):
            return 2
        backup = safeguard.snapshot([g for gs in globs.values() for g in gs], ROOT)
        if backup:
            print(f"copy of the files to be rewritten: {backup}", flush=True)
    print(f"{len(steps)} steps, {args.jobs} at a time, {threads} CPU threads each", flush=True)
    record_environment(out_dir if args.quick else ROOT / "data" / "results", logs, env, args.jobs, threads)

    pending, running, done, failed = list(steps), {}, set(), []
    t_all = time.time()
    while pending or running:
        for step in list(pending):
            name, stage, track = step
            if failed or len(running) >= args.jobs or (wait[name] and wait[name] not in done):
                continue
            pending.remove(step)
            log = open(logs / f"{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(command(stage), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**env, **({"TRACK": track} if track else {})})
            running[name] = (proc, log, time.time())
            print(f"{time.strftime('%H:%M:%S')}  start {name}", flush=True)
        for name, (proc, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            seconds, ok = time.time() - t0, proc.returncode == 0
            (done.add if ok else failed.append)(name)
            new = not times.exists()
            with open(times, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["step", "mode", "jobs", "threads", "quick", "started", "seconds", "status"])
                w.writerow([name, "scripts", args.jobs, threads, bool(args.quick),
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), round(seconds),
                            "ok" if ok else f"exit {proc.returncode}"])
            print(f"{time.strftime('%H:%M:%S')}  {'done' if ok else 'FAILED'} {name} in {seconds / 60:.1f} min "
                  f"(log: {logs / f'{name}.log'})", flush=True)
            if not ok and backup:
                restored, removed = safeguard.restore(backup, globs[name], ROOT)
                print(f"          {name}: {len(restored)} files put back as before the run, {len(removed)} new files removed",
                      flush=True)
        if failed and not running:
            break
        time.sleep(2)

    if failed:
        skipped = [n for n, _, _ in pending]
        print(f"failed: {', '.join(failed)}" + (f"; not started: {', '.join(skipped)}" if skipped else ""))
        return 1
    print(f"all done in {(time.time() - t_all) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
