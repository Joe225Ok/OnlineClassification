"""Stage 4: retraining on OOD, unseen sessions, whole flows.

Run it from anywhere (it works in the project folder), or run every stage with run_all.py in the project folder.
The track comes from the TRACK environment variable (A or B).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def show(value):
    """Print a table (or any value) left as the last expression of a block."""
    if value is None:
        return
    print(value.to_string() if hasattr(value, "to_string") else value)
    print()




# %% [markdown]
# # Stage 4: experiments (one track per run)
#
# Every experiment uses this track's own rows, cleaning rule, main model inputs and GRU size, and builds its deployable unit
# the same way as stage 3 (temperatures, OOD detectors and limits and precision-target cut-offs from its own validation data).
# All results are on test (or held-out) rows, in the "all" and "unseen" views ("unseen" on the key "all inputs", since
# every model here reads the main model's inputs).
#
# 1. **Retraining on OOD abstentions**: does labelling what the classifier flagged teach it something?
# 2. **Unseen sessions**: leave one capture out (7 runs), and train on February to May, test on August.
# 3. **Whole flows**: following each flow through the test period, memory reset vs carried across segments.
#
# Every model reads L = 50-packet segments; how early an answer can be given is shown by the decision packet K
# (1, 3, 5, 10, 20, 50) in every table, so no model is trained on another segment length.


# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import kitsune_lib as kl
from kitsune_lib import (CAPTURES, GRU_GRID, L, OOD_CAPTURES, SEED, build_unit, decide, fit_gru, fit_stats, group_columns,
                         known_metrics, load, load_data, load_gru, load_json, model_key, p_attack, predict, read_track,
                         route, rows_digest, split_index, time_split)

TRACK = os.environ.get("TRACK", "A")
OUT = kl.RESULT_DIR
HIDDEN_TRAIN = 30_000

X, segments, y_all, lengths, _, _ = load_data()
track = read_track(TRACK).set_index("segment", drop=False)
track["capture_order"] = track["capture"].map({c: i for i, c in enumerate(CAPTURES)})
part = split_index(track)
held = pd.read_parquet(kl.TRACK_DIR / "held_out.parquet").set_index("segment", drop=False)
selection = load_json(OUT / f"{TRACK}_selection.json")
arch, groups = selection["arch"], selection["final"]["main"]
cols = group_columns(groups)
seg_hash = segments[kl.KEYS[model_key(groups)]].to_numpy()  # every model here reads the main inputs: key "all inputs"
stats = fit_stats(X, part["train"])


def halves(idx, frame):
    """The validation split cut in two, stratified by capture and class: temperatures, then thresholds."""
    f = frame.loc[idx]
    strata = (f["capture"] + f["attack"].astype(str)).to_numpy()
    counts = pd.Series(strata).map(pd.Series(strata).value_counts()).to_numpy()
    strata = np.where(counts < 2, f["attack"].astype(str).to_numpy(), strata)  # a group of one: stratify by class only
    if pd.Series(strata).value_counts().min() < 2:
        strata = None
    return train_test_split(idx, test_size=0.5, random_state=SEED, stratify=strata)


def views(out, y, unseen_mask, packets=None):
    return {view: known_metrics(out[m], y[m], None if packets is None else packets[m])
            for view, m in [("all", np.ones(len(out), bool)), ("unseen", unseen_mask)]}


cal_idx, thr_idx = halves(part["val"], track)
parts = {"train sample": np.sort(np.random.default_rng(SEED).choice(part["train"], min(HIDDEN_TRAIN, len(part["train"])), replace=False)),
         "cal": np.sort(cal_idx), "thr": np.sort(thr_idx), "test": part["test"], "held": held.index.to_numpy()}
main_net = load_gru(selection["models"]["main"][0], len(cols), *GRU_GRID[arch],
                    rows=rows_digest(TRACK, part["train"], part["val"], cols, stats, segments))
u0 = build_unit(X, main_net, cols, stats, parts, lengths, y_all)
test_unseen = ~np.isin(seg_hash[part["test"]], seg_hash[part["train"]])
print(f"Track {TRACK}: size {arch}, inputs {groups}")


# %% [markdown]
# ## 1. Retraining on OOD abstentions
#
# Each held-out capture is cut in half by time. Early-half segments flagged OOD join the training set with their label, the
# GRU is retrained from scratch, and the unit is rebuilt. The late half, which neither model saw, shows what changed, and
# known test traffic checks that nothing else got worse.


# %%
hm = held.loc[parts["held"]]
early = (hm["start"] <= hm.groupby("capture")["start"].transform("median")).to_numpy()
out, _ = decide(u0, "held")
queued = parts["held"][early & (out == "OOD")]
print(f"early-half OOD abstentions added to training: {len(queued):,}", hm.loc[queued, "group"].value_counts().to_dict())
train_plus = np.r_[part["train"], queued]
net1, _ = fit_gru(f"{TRACK}_{arch}_retrained", load(X, train_plus, stats, cols), y_all[train_plus].astype(np.float32),
                  load(X, part["val"], stats, cols), y_all[part["val"]].astype(np.float32), *GRU_GRID[arch])
u1 = build_unit(X, net1, cols, stats, {**parts, "train sample": np.r_[parts["train sample"], queued]}, lengths, y_all)

rows = []
for name, u in [("before", u0), ("after", u1)]:
    out, packets = decide(u, "test")
    for view, r in views(out, u["y"]["test"], test_unseen, packets).items():
        rows.append({"model": name, "traffic": f"known test, {view}", **r})
    out, _ = decide(u, "held")
    for group in pd.unique(hm["group"]):
        for lab in (0, 1):
            m = (~early) & (hm["group"].to_numpy() == group) & (u["y"]["held"] == lab)
            o = out[m]
            rows.append({"model": name, "traffic": f"{group}, label {lab}, late half", "segments": int(m.sum()),
                         "failure rate": np.isin(o, ["High", "Low"]).mean() if m.any() else np.nan,
                         **{a: (o == a).mean() if m.any() else np.nan for a in ["High", "Low", "Medium", "OOD"]}})
retraining = pd.DataFrame(rows).set_index(["traffic", "model"]).sort_index()
retraining.to_csv(OUT / f"{TRACK}_retraining.csv")
show(retraining[["segments", "failure rate", "High", "Low", "Medium", "OOD", "attacks blocked", "block precision", "Low precision"]].round(4))


# %% [markdown]
# ## 2. Unseen sessions
#
# Each capture is its own recording, so a session can be held out whole, benign and attack together: the session and the
# attack are new, the network is the same. The training sessions' rows of this track are split 80/20 by time into training
# and validation; the held-out sessions' rows of this track are the test. The GRU size and inputs are this track's, chosen
# on data that included every session, which is a small remaining leak.


# %%
known = [c for c in CAPTURES if c not in OOD_CAPTURES]
SESSIONS = {f"without {c}": ([k for k in known if k != c], [c]) for c in known}
SESSIONS["Feb-May to Aug"] = (["Video Injection", "SSDP Flood", "SYN DoS", "SSL Renegotiation"], ["OS Scan", "Fuzzing", "Active Wiretap"])
rows = []
for run, (train_caps, test_caps) in SESSIONS.items():
    t0 = time.time()
    pool = track[track["capture"].isin(train_caps)].sort_values(["capture_order", "attack", "start"], kind="stable").copy()
    pool["split"] = pd.concat([time_split(g, (0.8,)) for _, g in pool.groupby(["capture", "attack"], sort=False)])
    tr = np.sort(pool.loc[pool["split"] == "train", "segment"].to_numpy())
    va = np.sort(pool.loc[pool["split"] == "val", "segment"].to_numpy())
    te = np.sort(track.loc[track["capture"].isin(test_caps), "segment"].to_numpy())
    st = fit_stats(X, tr)
    tag = "".join(ch for ch in run if ch.isalnum())
    net, _ = fit_gru(f"{TRACK}_{arch}_session_{tag}", load(X, tr, st, cols), y_all[tr].astype(np.float32),
                     load(X, va, st, cols), y_all[va].astype(np.float32), *GRU_GRID[arch])
    c, t = halves(va, pool)
    sp = {"train sample": np.sort(np.random.default_rng(SEED).choice(tr, min(HIDDEN_TRAIN, len(tr)), replace=False)),
          "cal": np.sort(c), "thr": np.sort(t), "test": te}
    u = build_unit(X, net, cols, st, sp, lengths, y_all)
    out, packets = decide(u, "test")
    for view, r in views(out, u["y"]["test"], ~np.isin(seg_hash[te], seg_hash[tr]), packets).items():
        rows.append({"run": run, "view": view, **r})
    print(f"{run}: {time.time() - t0:.0f}s")
sessions = pd.DataFrame(rows).set_index(["run", "view"])
sessions.to_csv(OUT / f"{TRACK}_sessions.csv")
show(sessions[["segments", "coverage", "attacks blocked", "block precision", "attacks called benign", "Low precision", "Medium", "OOD"]].round(4))


# %% [markdown]
# ## 3. Whole flows: memory reset vs carried
#
# Each flow is followed through this track's test period (its rows from the start of the test period of each capture), in
# order. **reset**: every segment starts from empty memory, as in training. **carried**: each segment starts from the
# hidden state the previous segment of the flow left. Same model and unit, not retrained; carried memory is an inference
# change only.


# %%
start_of_test = track.loc[part["test"]].groupby("capture")["start"].min()
stream = track[track["start"] >= track["capture"].map(start_of_test)].sort_values(["capture", "flow", "start"], kind="stable").reset_index(drop=True)
stream["flow_key"] = stream.groupby(["capture", "flow"], sort=False).ngroup()
stream["round"] = stream.groupby("flow_key").cumcount()
x_s = load(X, stream["segment"].to_numpy(), stats, cols)
n_s = stream["length"].to_numpy()
y_s = stream["attack"].to_numpy()
hidden, layers = GRU_GRID[arch]


@torch.no_grad()
def run_stream(carry):
    if not carry:
        return predict(main_net, x_s, with_hidden=True)
    z = np.zeros((len(stream), L), np.float32)
    h = np.zeros((len(stream), L, hidden), np.float32)
    state = torch.zeros(layers, stream["flow_key"].max() + 1, hidden)
    keys, rounds = stream["flow_key"].to_numpy(), stream["round"].to_numpy()
    for r in range(rounds.max() + 1):
        at = np.flatnonzero(rounds == r)
        o, last = main_net.gru(torch.from_numpy(x_s[at]), state[:, keys[at]])
        state[:, keys[at]] = last
        z[at], h[at] = main_net.head(o).squeeze(-1).numpy(), o.numpy()
    return z, h


t = u0["tuned"][L]
rows = []
for mode, carry in [("reset", False), ("carried", True)]:
    z, h = run_stream(carry)
    flags = kl.ood_flags(kl.ood_running_means(u0["detectors"], h, n_s), t["ood limits"])
    del h
    out, packets = route(p_attack(z, u0["temperatures"]), flags, n_s, t["tau_attack"], t["tau_benign"], L)
    d = stream[["flow_key", "attack", "length"]].assign(blocked=out == "High", low=out == "Low", packets=packets)
    d["before"] = d.groupby("flow_key")["length"].cumsum() - d["length"]
    per_flow = d.groupby("flow_key").agg(attack=("attack", "max"), blocked=("blocked", "max"), segments=("blocked", "size"))
    r = {"memory": mode, **known_metrics(out, y_s)}
    for name, f in [("all flows", per_flow), ("multi-segment flows", per_flow[per_flow["segments"] > 1])]:
        r[f"{name}: attack"] = int((f["attack"] == 1).sum())
        r[f"{name}: attack ever blocked"] = f.loc[f["attack"] == 1, "blocked"].mean()
        r[f"{name}: benign ever blocked"] = f.loc[f["attack"] == 0, "blocked"].mean()
    a = d[d["attack"] == 1]
    first_block = a[a["blocked"]].groupby("flow_key").first()
    passed = first_block["before"] + first_block["packets"] - a.groupby("flow_key")["before"].first().loc[first_block.index]
    r["attack packets before block, median"], r["attack packets before block, p95"] = passed.median(), passed.quantile(0.95)
    rows.append(r)
whole_flow = pd.DataFrame(rows).set_index("memory")
whole_flow.to_csv(OUT / f"{TRACK}_whole_flow.csv")
show(whole_flow.T.round(4))
