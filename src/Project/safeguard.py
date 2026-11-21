"""What a run overwrites, and how the old files are protected. Used by run_all.py.

Before the first step, the orchestrator lists every existing file the selected steps rewrite, states the consequences
and asks for yes or no. On yes it copies those files into runs/backup/<time>/ (a snapshot, a few tens of MB) and only
then starts. If a step fails, that step's files are restored from the snapshot and any file the failed step created is
removed, so its outputs are exactly as before the run. The snapshot stays until deleted by hand.

Trained models (data/models_v2/*.pt), the parsed data of stage 0 and the baseline probabilities are caches: a stage
writes them only when they are missing, so a run never overwrites them and they are not listed. A missing model is
retrained, which takes hours; the message says so.
"""
import shutil
import sys
import time
from pathlib import Path

# Files each stage rewrites, as globs relative to the project folder; {T} is the track. Caches written only when
# missing are deliberately absent. Appended logs (training_times.csv, run_times.csv) are not listed either.
REWRITES = {
    "s0_prepare": ["data/flows_vs_segments.csv", "data/results/iat_resolution.csv"],
    "s1_data": ["data/tracks/*.parquet", "data/results/diagnostics.csv", "data/results/conflicts_for_review.csv",
                "data/results/track_sizes.csv", "data/results/track_b_drops.csv", "data/figures/data_diagnostics.png"],
    "s2_models": ["data/results/{T}_gru_size.csv", "data/results/{T}_final_models.csv", "data/results/{T}_selection.json"],
    "s3_evaluate": ["data/models_v2/{T}_*_unit.joblib"] + [f"data/results/{{T}}_{n}" for n in (
        "thresholds.csv", "pipeline.csv", "bootstrap.csv", "per_attack_type.csv", "drift.csv", "decision_time.csv",
        "forced_and_ceilings.csv", "calibration.csv", "reliability.csv", "cap_answers.csv", "held_out.csv",
        "ood_auroc.csv", "ood_detectors.csv", "baselines.csv", "baselines_forced.csv", "testbed_comparison.csv",
        "evasion.csv", "deployment_cost.json")],
    "s4_experiments": ["data/results/{T}_retraining.csv", "data/results/{T}_sessions.csv", "data/results/{T}_whole_flow.csv"],
    "s5_stateful": ["data/results/{T}_stateful_forced.csv", "data/results/{T}_stateful_pipeline.csv",
                    "data/results/{T}_stateful_whole_flow.csv"],
    "export_artifacts": ["artifacts/{T}/*/*"],
    "s6_uncertainty": ["data/results/{T}_*_mc_*.csv"],
    "s7_monitoring": ["data/results/{T}_*_psi.csv"],
}

CONSEQUENCES = """\
What a rerun means:
  - The files above are recomputed and replaced: results tables, tracks, deployable units and exported artifacts.
    With the same trained models and the same number of CPU threads the values are identical; a different thread
    count can change the last digits.
  - Trained models in data/models_v2 and the parsed data of stage 0 are reused, never overwritten. A model whose file
    is missing is retrained, which takes hours.
  - Before the first step, every file listed is copied to {backup}. If a step fails, that step's files are put back
    from the copy and anything it created is removed. The copy stays until you delete it."""


def rewrites(script, track=None, extra=()):
    """The globs a step rewrites: those of its script with the track filled in, plus any the caller adds."""
    return [g.format(T=track) for g in REWRITES[script]] + list(extra)


def existing(globs, root):
    """The files under root that match the globs now, as paths relative to root, without duplicates."""
    found = {p.relative_to(root) for g in globs for p in Path(root).glob(g) if p.is_file()}
    return sorted(found, key=str)


def describe(steps, root, backup_name="runs/backup/<time>/"):
    """The message shown before a run: per step, the existing files it rewrites, then the consequences."""
    lines, total = [], 0
    for name, globs in steps:
        files = existing(globs, root)
        total += len(files)
        if files:
            shown = ", ".join(str(f).replace("\\", "/") for f in files[:3]) + (", ..." if len(files) > 3 else "")
            lines.append(f"  {name:22s} {len(files):3d} file{'s' if len(files) != 1 else ''}   {shown}")
    if not total:
        return ""
    head = f"This run rewrites {total} existing file{'s' if total != 1 else ''}:"
    return "\n".join([head, *lines, "", CONSEQUENCES.format(backup=backup_name)])


def confirm(message, assume_yes=False, ask=input):
    """Print the message and ask for yes or no. True to continue. No usable answer (no terminal) means no."""
    if not message:
        return True
    print(message, flush=True)
    if assume_yes:
        print("Continue? [yes/no] yes (given in advance)", flush=True)
        return True
    try:
        answer = ask("Continue? [yes/no] ").strip().lower()
    except (EOFError, OSError, RuntimeError, NotImplementedError) as e:  # no terminal to answer on
        print(f"no answer ({type(e).__name__}): aborted. Give the answer in advance to run unattended.", flush=True)
        return False
    if answer in ("y", "yes"):
        return True
    print("aborted, nothing was changed", flush=True)
    return False


def snapshot(globs, root, backup_root=None):
    """Copy every existing file matching the globs into a new folder under runs/backup, keeping the relative paths.
    Returns the folder, with a manifest of what it holds, or None when no file matches (nothing to protect)."""
    root = Path(root)
    files = existing(globs, root)
    if not files:
        return None
    folder = Path(backup_root or root / "runs" / "backup") / time.strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    for rel in files:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, folder / rel)
    (folder / "manifest.txt").write_text("\n".join(str(f) for f in files) + ("\n" if files else ""), encoding="utf-8")
    return folder


def restore(folder, globs, root):
    """Put a step's files back as they were in the snapshot: copy back every snapshotted file that matches the globs
    and remove any matching file that was not in the snapshot. Returns (restored, removed) as relative paths."""
    root, folder = Path(root), Path(folder)
    saved = {Path(line) for line in (folder / "manifest.txt").read_text(encoding="utf-8").splitlines() if line}
    wanted = {p.relative_to(folder) for g in globs for p in folder.glob(g) if p.is_file()} & saved
    restored, removed = [], []
    for rel in sorted(wanted, key=str):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / rel, root / rel)
        restored.append(rel)
    for rel in existing(globs, root):
        if rel not in saved:
            (root / rel).unlink()
            removed.append(rel)
    return restored, removed


if __name__ == "__main__":  # python safeguard.py: show what a full run of both tracks would rewrite, nothing else
    root = Path(__file__).resolve().parent
    steps = [(s + (f"_{t}" if t else ""), rewrites(s, t)) for s in REWRITES for t in ([None] if s in ("s0_prepare", "s1_data") else ["A", "B"])]
    print(describe(steps, root) or "nothing to rewrite")
    sys.exit(0)
