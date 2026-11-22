"""What the integration test and tests/record_reference.py share: running every pipeline script, unchanged, on a crafted
archive in a temporary folder, and reading the values compared with the recorded reference.

Only four things differ from a real run:
  - the data: a small crafted archive (tests/edge_case_data.py) instead of the 19 GB one;
  - the training budget: tiny GRUs, at most 8 epochs, patience 2, batches of 32, learning rate 0.02 (kitsune_lib's
    constants, patched while the pipeline runs);
  - one thread, whatever the machine: PyTorch and the BLAS and OpenMP pools of numpy, scipy and scikit-learn. The
    number of threads changes the order in which sums are added, so the last digits of the results. On the tiny
    models of this test, that is enough to move a borderline segment across an OOD limit, and the reference values
    would then depend on how many cores the machine has. Stage 3 normally sets PyTorch to every core; here it cannot;
  - os.chdir is a no-op while a script runs: each script starts with os.chdir(<project folder>), and must never write
    into the real data/ folder from a test. The export and inference are pointed at the temporary folder as well.
"""
import contextlib
import math
import os
import platform
import runpy
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_limits

import kitsune_lib as kl

PROJECT = Path(kl.__file__).resolve().parent
PIPELINE = PROJECT / "pipeline"
ARCHIVE = "kitsune+network+attack+dataset.zip"
# One reference per operating system: PyTorch and the math libraries run different low-level code on Windows and Linux,
# and on these tiny models that is enough to move a few borderline segments (one thread removes only the differences
# that come from the number of cores). Each file is recorded on its own platform by tests/record_reference.py from
# the same agreed code. Linux: recorded on the GitHub runner (.github/workflows/record-reference.yml).
REFERENCE = Path(__file__).resolve().parent / "reference" / f"integration_reference_{platform.system().lower()}.json"
REAL_CHDIR = os.chdir  # kept before anything replaces os.chdir
STAGES = ["s0_prepare", "s1_data", "s2_models", "s3_evaluate", "s4_experiments", "s5_stateful", "export_artifacts",
          "infer", "s6_uncertainty", "s7_monitoring"]
NO_TESTBED = "without TTL, destination port and IAT"


class Pipeline:
    def __init__(self, root, spec):
        self.root, self.spec, self.ran = root, spec, []

    def run(self, stage, run_name="__main__"):
        """Run pipeline/<stage>.py in the temporary folder and return its namespace."""
        assert Path.cwd() == self.root, "a stage must only ever run in the temporary folder"
        ns = runpy.run_path(str(PIPELINE / f"{stage}.py"), run_name=run_name)
        self.ran.append(stage)
        return ns

    def result(self, name, **kw):
        return pd.read_csv(self.root / "data" / "results" / name, **kw)

    def json(self, name):
        return kl.load_json(self.root / "data" / "results" / name)

    def model_name(self, role):
        return self.json("A_selection.json")["models"][role][0]

    def run_everything(self):
        """Every stage for Track A, the export into the temporary folder, and infer.py on Mirai's raw capture."""
        for stage in STAGES:
            try:
                if stage == "export_artifacts":  # not __main__: that would write into the real artifacts/ folder
                    self.run(stage, run_name="export_artifacts")["export"]("A", out_root=self.root / "artifacts")
                elif stage == "infer":
                    self.infer_mirai()
                else:
                    self.run(stage)
            except BaseException as e:  # SystemExit included: name the stage that stopped
                raise RuntimeError(f"pipeline stage {stage} failed: {e!r}") from e

    def infer_mirai(self):
        """infer.py on Mirai's raw capture and labels, as a deployment would run it, with the exported model."""
        with zipfile.ZipFile(self.root / "data" / ARCHIVE) as z:
            for n in z.namelist():
                if n.startswith("mirai/"):
                    (self.root / Path(n).name).write_bytes(z.read(n))
        argv = sys.argv
        sys.argv = ["infer.py", str(self.root / "Mirai_pcap.pcap.gz"), "--labels", str(self.root / "Mirai_labels.csv.gz"),
                    "--artifacts", str(self.root / "artifacts"), "--out", str(self.root / "mirai_answers.csv")]
        try:
            self.run("infer")
        finally:
            sys.argv = argv


@contextlib.contextmanager
def isolated_pipeline(root, write_archive):
    """`write_archive(path)` writes the archive (and returns its specification); the working directory is `root` and the
    scripts stay there; everything is undone on exit."""
    for d in ("data/tracks", "data/models_v2", "data/results"):
        (root / d).mkdir(parents=True)
    spec = write_archive(root / "data" / ARCHIVE)
    before, threads = Path.cwd(), torch.get_num_threads()
    REAL_CHDIR(root)
    try:
        with pytest.MonkeyPatch.context() as mp, threadpool_limits(limits=1):
            torch.set_num_threads(1)
            mp.setattr(torch, "set_num_threads", lambda *_: None)  # one thread whatever the machine (see the top)
            mp.setattr(os, "chdir", lambda *_: None)
            mp.setenv("TRACK", "A")
            mp.setenv("ROLE", "no_testbed")
            mp.setenv("MPLBACKEND", "Agg")
            mp.setattr(kl, "GRU_GRID", {"64": (4, 1), "128": (8, 1), "128x2": (8, 2)})
            mp.setattr(kl, "MAX_EPOCHS", 8)
            mp.setattr(kl, "PATIENCE", 2)
            mp.setattr(kl, "BATCH", 32)
            mp.setattr(kl, "LR", 0.02)  # a few epochs on little data need larger steps
            yield Pipeline(root, spec)
    finally:
        REAL_CHDIR(before)
        torch.set_num_threads(threads)  # stage 3 sets it to every core


# --- Values compared with the recorded reference ---------------------------------------------------------------------
# EXACT: depend only on the data and the splits (identical on every platform). RATE: shares and probabilities produced
# by trained models, within 0.05. SCALE: unbounded model outputs (losses, PSI of model outputs), within 25% or 0.05.
EXACT, RATE, SCALE = {"abs": 1e-9}, {"abs": 0.05}, {"rel": 0.25, "abs": 0.05}

# Machine-sensitive values: shares of few segments that moved by more than half their tolerance (0.025) from
# machine-level rounding alone, on the same code, data and SEED. Measured on 8 to 11 runs: the recorded references,
# six forced CPU code paths (PyTorch, OpenBLAS, MKL) and GitHub's Linux runners. Spread seen: Mirai 0.10 (5 of 50
# segments); sessions without SSDP Flood 0.088, OS Scan 0.074, SSL Renegotiation 0.074, SYN DoS 0.059, Fuzzing 0.059,
# Video Injection 0.029 (of 136 segments). Every other value moved by at most 0.015, most not at all. They keep their
# tolerance. With SKIP_MACHINE_SENSITIVE=1 (set in CI, whose runners differ in CPU), they are skipped, with the reason
# shown. Everywhere else they are compared as usual, so on the machine that recorded the reference they are checked.
MACHINE_SENSITIVE = frozenset({
    "s3 main: held-out Mirai, before infection, OOD",
    *(f"s4 session without {c}: coverage" for c in ("SSDP Flood", "OS Scan", "SSL Renegotiation", "SYN DoS", "Fuzzing",
                                                    "Video Injection")),
})


def number(v):
    v = float(v)
    return None if math.isnan(v) else v


def collect(p):
    """{name: (value, tolerance)} for the reference comparison, read from the outputs of a finished run."""
    out, main, nt = {}, p.model_name("main"), p.model_name(NO_TESTBED)

    def put(name, value, tol):
        out[name] = (value if isinstance(value, str) else number(value), tol)

    # stage 2: the size chosen and its validation losses
    sizes = p.result("A_gru_size.csv", index_col=0)
    put("s2 chosen size", p.json("A_selection.json")["arch"], EXACT)
    for arch, loss in sizes["validation loss, all"].items():
        put(f"s2 validation loss, size {arch}", loss, SCALE)
    # stage 3: cut-offs and test results at K = 50, both models
    th = p.result("A_thresholds.csv").set_index(["model", "K"])
    pipe = p.result("A_pipeline.csv").set_index(["model", "split", "K", "view"])
    for role, name in (("main", main), ("no testbed", nt)):
        for col in ("tau_attack", "tau_benign"):
            put(f"s3 {role}: {col}, K=50", th.loc[(name, 50), col], RATE)
        for col in ("coverage", "attacks blocked", "block precision", "Low precision", "Medium", "OOD"):
            put(f"s3 {role}: test {col}, K=50", pipe.loc[(name, "test", 50, "all"), col], RATE)
    forced = p.result("A_forced_and_ceilings.csv").set_index(["model", "split", "view", "k"])
    for k in (1, 3, 50):
        put(f"s3 no testbed: MaxAcc (headers), test, k={k}", forced.loc[(nt, "test", "all", k), "MaxAcc, headers"], EXACT)
        put(f"s3 no testbed: forced accuracy, test, k={k}", forced.loc[(nt, "test", "all", k), "accuracy"], RATE)
    for k in (1, 50):
        put(f"s3 main: MaxAcc (all inputs), test, k={k}", forced.loc[(main, "test", "all", k), "MaxAcc, all inputs"], EXACT)
        put(f"s3 main: forced accuracy, test, k={k}", forced.loc[(main, "test", "all", k), "accuracy"], RATE)
    cal = p.result("A_calibration.csv").set_index(["model", "temperature", "view", "k"])
    put("s3 no testbed: ECE, capped, test, k=50", cal.loc[(nt, "capped (used)", "all", 50), "ECE"], RATE)
    held = p.result("A_held_out.csv")
    held = held[(held["model"] == main) & (held["view"] == "all")].set_index("traffic")
    for traffic in held.index:
        put(f"s3 main: held-out {traffic}, OOD", held.loc[traffic, "OOD"], RATE)
    base = p.result("A_baselines.csv").set_index(["model", "view"])
    for m in ("GRU", "k-NN", "XGBoost"):
        put(f"s3 baseline {m}: coverage", base.loc[(m, "all"), "coverage"], RATE)
    ev = p.result("A_evasion.csv").set_index("perturbation")
    for pert in ev.index:
        put(f"s3 evasion {pert}: High", ev.loc[pert, "High"], RATE)
    # stage 4: sessions, retraining, whole flows
    sess = p.result("A_sessions.csv").set_index(["run", "view"])
    for run in sess.index.get_level_values(0).unique():
        put(f"s4 session {run}: coverage", sess.loc[(run, "all"), "coverage"], RATE)
    rt = p.result("A_retraining.csv").set_index(["traffic", "model"])
    for model in ("before", "after"):
        put(f"s4 retraining {model}: known test coverage", rt.loc[("known test, all", model), "coverage"], RATE)
    wf = p.result("A_whole_flow.csv", index_col=0)
    for mode in wf.index:
        put(f"s4 whole flows, memory {mode}: coverage", wf.loc[mode, "coverage"], RATE)
    # stage 5: the stateful model
    st = p.result("A_stateful_pipeline.csv").set_index(["split", "view", "K", "memory"])
    for mode in ("reset", "carried"):
        put(f"s5 memory {mode}: test coverage, K=50", st.loc[("test", "all", 50, mode), "coverage"], RATE)
    # stage 6: MC dropout
    mc = p.result("A_no_testbed_mc_pipeline.csv").set_index(["reading", "K", "view"])
    for reading in ("deterministic", "MC average"):
        put(f"s6 {reading}: test coverage, K=50", mc.loc[(reading, 50, "all"), "coverage"], RATE)
    u = p.result("A_no_testbed_mc_uncertainty_by_answer.csv", header=[0, 1], index_col=[0, 1])
    for part in ("aleatoric", "epistemic"):
        put(f"s6 known test traffic: mean {part} (bits)", np.average(u[(part, "mean")], weights=u[("segments", "Unnamed: 2_level_1")]), RATE)
    # stage 7: PSI. Input fields depend only on the data (exact); model outputs on the model
    psi = p.result("A_no_testbed_psi.csv", index_col=[0, 1])
    for comp in ("Feb-May captures -> Aug captures", "training -> held-out Mirai"):
        for field in ("frame_len", "tcp_window", "dst_port"):
            put(f"s7 PSI {comp}, benign: {field}", psi.loc[(comp, "benign"), field], EXACT)
        for field in ("P(attack)", "answer mix"):
            put(f"s7 PSI {comp}, all: {field}", psi.loc[(comp, "all"), field], SCALE)
    return out
