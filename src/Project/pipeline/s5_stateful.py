"""Stage 5: stateful GRU against the main model.

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
# # Stage 5: stateful GRU (one track per run)
#
# So far every segment starts from empty memory, so packet 51 of a flow looks like new traffic. Here the GRU is **trained**
# to carry its hidden state from one segment of a flow to the next; gradients stop at each segment boundary, and a new flow
# starts empty. The **reset** model is this track's main model (stage 2): same inputs, size, data and loss, without memory.
#
# - **Chains**: every known segment in flow order, so memory is never broken by a missing segment. For Track B, conflicting
#   segments pass memory along but never count in the loss or the scores.
# - **Training**: the track's training segments, each starting from the saved state of the previous training segment of its
#   flow. Early stopping on the validation segments, read in flow order with memory carried through training and validation.
# - **Reading**: on validation and test, memory runs through every earlier segment of the flow, as it would in deployment.
# - **Decisions**: temperatures on the first validation half, precision-target cut-offs on the second, OOD gate off for
#   both models so that only memory differs. Results on validation and test, in the "all" and "unseen" views, plus whole flows.


# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn as tnn

import kitsune_lib as kl
from kitsune_lib import (BATCH, CAPTURES, GRU_GRID, K_REPORT, L, MAX_EPOCHS, PATIENCE, SEED, OnlineGRU, ceiling, fit_gru,
                         fit_stats, fit_temperatures, group_columns, known_metrics, load, load_data, load_json, p_attack,
                         read_track, real_steps, route, scores, at_prefix, tune_targets)

TRACK = os.environ.get("TRACK", "A")
OUT = kl.RESULT_DIR
QUICK = os.environ.get("QUICK")
if QUICK:
    kl.MODEL_DIR = OUT = Path(QUICK)
    MAX_EPOCHS = 1

X, segments, y_all, lengths, _, _ = load_data()
seg_hash = segments["hash"].to_numpy()
selection = load_json(OUT / f"{TRACK}_selection.json")
arch, groups = selection["arch"], selection["final"]["main"]
cols = group_columns(groups)
hidden, layers = GRU_GRID[arch]
mask_at = len(cols) - 1

# Chains: every known segment (Track A keeps them all), in flow order; this track's split, or none if not in the track
chain = read_track("A").drop(columns="split").merge(read_track(TRACK)[["segment", "split"]], on="segment", how="left")
chain["split"] = chain["split"].fillna("not in track")
chain = chain.sort_values(["capture", "flow", "start"], kind="stable").reset_index(drop=True)
if QUICK:
    keep_flows = chain.groupby(["capture", "flow"]).ngroup() % 100 == 0
    chain = chain[keep_flows].reset_index(drop=True)
chain["flow_key"] = chain.groupby(["capture", "flow"], sort=False).ngroup()
split = chain["split"].to_numpy()
seg = chain["segment"].to_numpy()
keys = chain["flow_key"].to_numpy()
y = chain["attack"].to_numpy()
n = chain["length"].to_numpy()
train_rows = np.flatnonzero(split == "train")
stats = fit_stats(X, np.sort(seg[train_rows]))
x = load(X, seg, stats, cols)
weight = float((y[train_rows] == 0).sum() / (y[train_rows] == 1).sum())
print(f"Track {TRACK}: {len(chain):,} chained segments in {chain['flow_key'].nunique():,} flows;",
      chain["split"].value_counts().to_dict())


# %% [markdown]
# ## Training with memory carried


# %%
def previous_in(rows):
    """For each of `rows` (in chain order), the position in `rows` of the previous one of its flow, or len(rows)."""
    prev = np.r_[len(rows), np.arange(len(rows) - 1)]
    prev[np.r_[True, keys[rows][1:] != keys[rows][:-1]]] = len(rows)
    return prev


@torch.no_grad()
def read_flows(net, rows, carry=True):
    """Logits of `rows` (in chain order), each segment starting from the state its flow's previous row left."""
    if not carry:
        return kl.predict(net, x[rows])
    net.eval()
    z = np.zeros((len(rows), L), np.float32)
    rounds = pd.Series(keys[rows]).groupby(keys[rows]).cumcount().to_numpy()
    state = torch.zeros(layers, keys.max() + 1, hidden)
    for r in range(rounds.max() + 1):
        at = np.flatnonzero(rounds == r)
        o, last = net.gru(torch.from_numpy(x[rows[at]]), state[:, keys[rows[at]]])
        state[:, keys[rows[at]]] = last
        z[at] = net.head(o).squeeze(-1).numpy()
    return z


def weighted_loss(z, rows):
    """The training loss (class-weighted, real packets only) on logits z of `rows`."""
    real = real_steps(n[rows])
    benign = (1 - y[rows])[:, None]
    w = real * (1 + (weight - 1) * y[rows][:, None])
    return float(((np.logaddexp(0, z) - benign * z) * w).sum() / w.sum())


def train_stateful(name):
    path = kl.MODEL_DIR / f"{name}.pt"
    torch.manual_seed(SEED)
    net = OnlineGRU(len(cols), hidden, layers)
    if path.exists():
        net.load_state_dict(torch.load(path)["state"])
        return net
    prev = previous_in(train_rows)
    store = torch.zeros(layers, len(train_rows) + 1, hidden)  # last column stays zero: the start of a flow
    read_rows = np.flatnonzero(np.isin(split, ["train", "val"]))  # validation read after its flow's training segments
    val_at = split[read_rows] == "val"
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    best, best_state, stale, t0 = np.inf, None, 0, time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(len(train_rows)), len(train_rows) // BATCH + 1):
            rows = train_rows[b]
            steps = int(n[rows].max())
            xb, yb = torch.from_numpy(x[rows, :steps]), torch.from_numpy(y[rows].astype(np.float32))
            o, last = net.gru(xb, store[:, prev[b]])
            store[:, b] = last.detach()
            logits = net.head(o).squeeze(-1)
            w = xb[:, :, mask_at] * (1 + (weight - 1) * yb.unsqueeze(1))
            loss = (tnn.functional.binary_cross_entropy_with_logits(logits, (1 - yb).unsqueeze(1).expand_as(logits),
                                                                    reduction="none") * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        val_loss = weighted_loss(read_flows(net, read_rows)[val_at], read_rows[val_at])
        print(f"{name}: epoch {epoch + 1}, validation loss {val_loss:.4f}, {time.time() - t0:.0f}s", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    net.load_state_dict(best_state)
    seconds = time.time() - t0
    torch.save({"state": best_state, "val_loss": best, "weight": weight, "epochs": epoch + 1, "seconds": seconds}, path)
    times = kl.MODEL_DIR / "training_times.csv"
    pd.DataFrame([{"model": name, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "epochs": epoch + 1, "seconds": round(seconds),
                   "training segments": len(train_rows), "hidden": hidden, "layers": layers}]
                 ).to_csv(times, mode="a", header=not times.exists(), index=False)
    return net


stateful = train_stateful(f"{TRACK}_{arch}_stateful")
main_net, _ = fit_gru(selection["models"]["main"][0], np.zeros((1, 1, len(cols)), np.float32), np.zeros(1, np.float32),
                      None, None, hidden, layers)


# %% [markdown]
# ## Results: reset (main model) vs carried (stateful)
#
# Both models are read over the same chains; only the memory differs. Every row reported is one of this track's segments.


# %%
val_rows, test_rows = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
f = chain.iloc[val_rows]
strata = (f["capture"] + f["attack"].astype(str)).to_numpy()
counts = pd.Series(strata).map(pd.Series(strata).value_counts()).to_numpy()
strata = np.where(counts < 2, f["attack"].astype(str).to_numpy(), strata)  # a group of one: stratify by class only
if pd.Series(strata).value_counts().min() < 2:
    strata = None
cal_rows, thr_rows = (np.sort(r) for r in train_test_split(val_rows, test_size=0.5, random_state=SEED, stratify=strata))
train_hashes = seg_hash[seg[train_rows]]
everything = np.arange(len(chain))

forced, pipeline, flows = [], [], []
for mode, net, carry in [("reset", main_net, False), ("carried", stateful, True)]:
    z_all = read_flows(net, everything, carry)  # memory runs through every earlier segment of the flow
    T, _ = fit_temperatures(z_all[cal_rows], n[cal_rows], y[cal_rows])
    p_all = p_attack(z_all, T)
    no_flags = np.zeros_like(p_all, bool)
    for part_name, rows in [("validation", val_rows), ("test", test_rows)]:
        unseen_m = ~np.isin(seg_hash[seg[rows]], train_hashes)
        for view, m in [("all", np.ones(len(rows), bool)), ("unseen", unseen_m)]:
            r = rows[m]
            for k in [1, 5, 20, 50]:
                pred = (at_prefix(z_all[r], n[r], k) < 0).astype(int)
                c = ceiling(seg_hash[seg[r]], y[r])
                forced.append({"memory": mode, "split": part_name, "view": view, "k": k, **scores(y[r], pred),
                               "MaxAcc (same rows)": c["MaxAcc"], "MaxF1 (same rows)": c["MaxF1"]})
    for K in K_REPORT:
        t = tune_targets(p_all[thr_rows], no_flags[thr_rows], n[thr_rows], y[thr_rows], K)
        out, packets = route(p_all, no_flags, n, t["tau_attack"], t["tau_benign"], K)
        for part_name, rows in [("validation", val_rows), ("test", test_rows)]:
            unseen_m = ~np.isin(seg_hash[seg[rows]], train_hashes)
            for view, m in [("all", np.ones(len(rows), bool)), ("unseen", unseen_m)]:
                r = rows[m]
                pipeline.append({"memory": mode, "split": part_name, "view": view, "K": K, "tau_attack": t["tau_attack"],
                                 "tau_benign": t["tau_benign"], **known_metrics(out[r], y[r], packets[r])})
        if K == L:  # whole flows through the test period, at the working cut-off
            d = chain.iloc[test_rows][["flow_key", "attack", "length"]].assign(blocked=out[test_rows] == "High",
                                                                                 packets=packets[test_rows])
            d["before"] = d.groupby("flow_key")["length"].cumsum() - d["length"]
            per_flow = d.groupby("flow_key").agg(attack=("attack", "max"), blocked=("blocked", "max"), segments=("blocked", "size"))
            row = {"memory": mode}
            for name, fl in [("all flows", per_flow), ("multi-segment flows", per_flow[per_flow["segments"] > 1])]:
                row[f"{name}: attack"] = int((fl["attack"] == 1).sum())
                row[f"{name}: benign"] = int((fl["attack"] == 0).sum())
                row[f"{name}: attack ever blocked"] = fl.loc[fl["attack"] == 1, "blocked"].mean()
                row[f"{name}: benign ever blocked"] = fl.loc[fl["attack"] == 0, "blocked"].mean()
            a = d[d["attack"] == 1]
            first_block = a[a["blocked"]].groupby("flow_key").first()
            passed = first_block["before"] + first_block["packets"] - a.groupby("flow_key")["before"].first().loc[first_block.index]
            row["attack packets before block, median"] = passed.median()
            row["attack packets before block, p95"] = passed.quantile(0.95)
            flows.append(row)

forced = pd.DataFrame(forced).set_index(["split", "view", "k", "memory"]).sort_index()
pipeline = pd.DataFrame(pipeline).set_index(["split", "view", "K", "memory"]).sort_index()
flows = pd.DataFrame(flows).set_index("memory")
forced.to_csv(OUT / f"{TRACK}_stateful_forced.csv")
pipeline.to_csv(OUT / f"{TRACK}_stateful_pipeline.csv")
flows.to_csv(OUT / f"{TRACK}_stateful_whole_flow.csv")
show((forced.loc["test"] * 100).round(2))


# %%
shown = ["segments", "coverage", "attacks blocked", "block precision", "attacks called benign", "Low precision",
         "accuracy when answering", "macro F1 when answering", "Medium", "mean packets to block"]
show(pipeline.xs(L, level="K")[shown].round(4))


# %%
show(flows.T.round(4))
