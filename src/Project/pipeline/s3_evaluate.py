"""Stage 3: deployable units, thresholds, views, ceilings, baselines and the real-world checks.

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
# # Stage 3: evaluation (one track per run)
#
# The track comes from `TRACK`. Each final model from stage 2 becomes one **deployable unit**, built only from this track's
# data: input statistics (training), temperatures (first validation half), two OOD detectors on training hidden states
# (an Isolation Forest, and the Mahalanobis distance to the nearer class mean with a shared covariance), OOD limits and
# the High/Low cut-offs (second validation half). Test and the held-out captures are used only to measure.
#
# **Decision rule**, per segment, packet by packet up to K (K = L = 50 is the working cut-off; shorter K are reported):
# OOD if the running mean of either detector's score is over its limit (forest OR Mahalanobis); High (block) if
# P(attack) >= tau_attack; Low (benign, this segment only) at the segment's end if P(attack) <= tau_benign; otherwise
# Medium ("I don't know").
#
# **Thresholds** are per-track precision targets on the second validation half: tau_attack is the lowest cut-off with at
# least 99% of blocks real attacks, tau_benign the highest with at least 99.9% of Low answers really benign. The two OOD
# limits sit at the same quantile of their detector's known validation scores, the lowest at which at most 5% of known
# validation segments are ever flagged by either detector.
#
# **Views**: "all" is the split as it is; "unseen" is "all" with every row whose sequence appears in this track's training
# split removed. Ceilings (MaxAcc: Jerabek et al. Eq. 1; minimum FPR: their lower bound; MaxF1 and the oracle's missed attacks: our additions) are computed on
# exactly the rows each result is measured on.


# %%
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import kitsune_lib as kl
from kitsune_lib import (BUCKETS, CAPTURES, GRU_GRID, K_REPORT, L, SEED, STEP_BUCKET, at_prefix, at_rate, ceiling,
                         fit_gru, fit_stats, fit_temperatures, flow_bootstrap, group_columns, known_metrics, load,
                         load_data, load_json, ood_thresholds, p_attack, predict, read_track, real_steps, route,
                         running_mean, scores, split_index, tune_targets)

TRACK = os.environ.get("TRACK", "A")
OUT = kl.RESULT_DIR
QUICK = os.environ.get("QUICK")
if QUICK:  # smoke test against the stage-2 smoke-test models
    kl.MODEL_DIR = OUT = Path(QUICK)
HIDDEN_TRAIN, FOREST_STEPS = 30_000, 200_000
DEPLOY_RATE = 0.01  # attack share assumed for the "real network" precision line

X, segments, y_all, lengths, prefix_header, prefix_timed = load_data()
track = read_track(TRACK).set_index("segment", drop=False)
part = split_index(track)
held = pd.read_parquet(kl.TRACK_DIR / "held_out.parquet").set_index("segment", drop=False)
if QUICK:
    part = {k: v[::max(1, len(v) // 3000)] for k, v in part.items()}
    held = held.iloc[::max(1, len(held) // 3000)]
seg_hash = segments["hash"].to_numpy()
train_hashes = seg_hash[part["train"]]
selection = load_json(OUT / f"{TRACK}_selection.json")
arch = selection["arch"]
print(f"Track {TRACK}: size {arch}, main groups: {selection['final']['main']}")

# The two validation halves, stratified by capture and class
from sklearn.model_selection import train_test_split
v = track.loc[part["val"]]
strata = (v["capture"] + v["attack"].astype(str)).to_numpy()
counts = pd.Series(strata).map(pd.Series(strata).value_counts()).to_numpy()
strata = np.where(counts < 2, v["attack"].astype(str).to_numpy(), strata)  # a group of one: stratify by class only
cal_idx, thr_idx = train_test_split(part["val"], test_size=0.5, random_state=SEED,
                                    stratify=strata if pd.Series(strata).value_counts().min() >= 2 else None)
parts = {"train sample": np.sort(np.random.default_rng(SEED).choice(part["train"], min(HIDDEN_TRAIN, len(part["train"])), replace=False)),
         "cal": np.sort(cal_idx), "thr": np.sort(thr_idx), "val": part["val"], "test": part["test"], "held": held.index.to_numpy()}
unseen_of = {p: ~np.isin(seg_hash[i], train_hashes) for p, i in parts.items()}
print({p: f"{len(i):,} ({unseen_of[p].mean():.0%} unseen)" for p, i in parts.items()})


# %% [markdown]
# ## Deployable units
#
# `build_unit` builds everything a deployment of one model needs and saves it. The main model uses all feature
# groups; the model without TTL, destination port and IAT is built the same way.


# %%
stats = fit_stats(X, part["train"])


def unit_for(model_name, groups):
    cols = group_columns(groups)
    model, _ = fit_gru(model_name, np.zeros((1, 1, len(cols)), np.float32), np.zeros(1, np.float32), None, None, *GRU_GRID[arch])
    u = kl.build_unit(X, model, cols, stats, parts, lengths, y_all, forest_steps=FOREST_STEPS)
    u.update(model=model_name, groups=groups)
    joblib.dump({k: u[k] for k in ["model", "groups", "columns", "stats", "temperatures", "fitted temperatures", "detectors", "thresholds"]},
                kl.MODEL_DIR / f"{model_name}_unit.joblib")
    return u


units = {}
for role, names in selection["models"].items():
    for name in names:
        t0 = time.time()
        units[name] = unit_for(name, selection["final"][role])
        print(f"{name}: unit built in {time.time() - t0:.0f}s")
main = units[selection["models"]["main"][0]]
thresholds = pd.DataFrame({(name, K): {k: v for k, v in t.items() if k != "ood limits"} for name, u in units.items()
                           for K, t in u["thresholds"].items()}).T
thresholds.index.names = ["model", "K"]
thresholds.to_csv(OUT / f"{TRACK}_thresholds.csv")
show(thresholds.xs(L, level="K"))


# %% [markdown]
# ## Known traffic: pipeline results, "I know" performance, all and unseen


# %%
def decide(u, p, K):
    t = u["tuned"][K]
    return route(u["prob"][p], t["flags"][p], u["n"][p], t["tau_attack"], t["tau_benign"], K)


rows = []
for name, u in units.items():
    for p in ("val", "test"):
        for K in K_REPORT:
            out, packets = decide(u, p, K)
            for view, m in [("all", np.ones(len(out), bool)), ("unseen", unseen_of[p])]:
                r = known_metrics(out[m], u["y"][p][m], packets[m])
                r["block precision at 1% attacks"] = at_rate(r["attacks blocked"], r["benign blocked"], DEPLOY_RATE)
                rows.append({"model": name, "split": p, "K": K, "view": view, **r})
pipeline = pd.DataFrame(rows).set_index(["model", "split", "K", "view"])
pipeline.to_csv(OUT / f"{TRACK}_pipeline.csv")
shown = ["segments", "coverage", "attacks blocked", "block precision", "block precision at 1% attacks", "attacks called benign",
         "benign blocked", "benign called Low", "Low precision", "accuracy when answering", "macro F1 when answering",
         "Medium", "OOD", "mean packets to block"]
show(pipeline.loc[(main["model"], ["val", "test"], L), shown].round(4))


# %%
# 95% intervals on test, resampling whole flows (segments of one flow are not independent)
flow_key = (track.loc[parts["test"], "capture"] + "/" + track.loc[parts["test"], "flow"].astype(str)).to_numpy()
out, _ = decide(main, "test", L)
cis = {view: flow_bootstrap(out[m], main["y"]["test"][m], flow_key[m]) for view, m in
       [("all", np.ones(len(out), bool)), ("unseen", unseen_of["test"])]}
pd.DataFrame(cis).to_csv(OUT / f"{TRACK}_bootstrap.csv")
show(pd.DataFrame(cis))


# %% [markdown]
# ## Per attack type, with the average across types


# %%
capture_of = track.loc[parts["test"], "capture"].to_numpy()
start_of = track.loc[parts["test"], "start"].to_numpy()
out, packets = decide(main, "test", L)
rows = []
for view, vm in [("all", np.ones(len(out), bool)), ("unseen", unseen_of["test"])]:
    for capture in CAPTURES:
        m = vm & (capture_of == capture)
        if not m.any():
            continue
        r = known_metrics(out[m], main["y"]["test"][m], packets[m])
        hours = max((start_of[capture_of == capture].max() - start_of[capture_of == capture].min()) / 3600, 1e-9)
        y_m, o_m = main["y"]["test"][m], out[m]
        r.update({"test hours": hours, "false blocks per hour": ((o_m == "High") & (y_m == 0)).sum() / hours,
                  "Medium per hour": (o_m == "Medium").sum() / hours,
                  "attacks passed per hour": ((o_m != "High") & (y_m == 1)).sum() / hours})
        rows.append({"view": view, "attack type": capture, **r})
per_type = pd.DataFrame(rows).set_index(["view", "attack type"])
for view in ["all", "unseen"]:  # every attack type weighs the same in the average
    per_type.loc[(view, "average across types"), :] = per_type.loc[view].mean(numeric_only=True)
per_type.to_csv(OUT / f"{TRACK}_per_attack_type.csv")
show(per_type[["attack segments", "attacks blocked", "attacks called benign", "Medium", "block precision", "Low precision",
          "false blocks per hour", "attacks passed per hour"]].round(4))


# %%
# Drift: four time slices of each capture's test period
rank = pd.Series(start_of).groupby(capture_of).rank(pct=True).to_numpy()
slice_of = np.minimum((rank * 4).astype(int), 3) + 1
drift = pd.DataFrame([{"slice": s, **known_metrics(out[slice_of == s], main["y"]["test"][slice_of == s])} for s in range(1, 5)]).set_index("slice")
drift.to_csv(OUT / f"{TRACK}_drift.csv")
show(drift[["segments", "coverage", "attacks blocked", "block precision", "Low precision", "Medium", "OOD"]].round(4))


# %% [markdown]
# ## Time to a decision
#
# How much traffic time passes before an answer: the seconds between a segment's first packet and the packet its answer
# came at (from the packets' inter-arrival times), for each decision packet K. "Blocks" are High answers on attack
# segments; "any answer" covers every segment. Compute time is separate (deployment cost below).


# %%
iat_s = np.expm1(np.array(X[parts["test"], :, kl.COL["iat"]], dtype=np.float64)) / 1e6  # seconds since the flow's previous packet
iat_s[:, 0] = 0  # counted from the segment's first packet
elapsed = np.cumsum(iat_s, axis=1)
rows = []
for name, u in units.items():
    y = u["y"]["test"]
    for K in K_REPORT:
        out, packets = decide(u, "test", K)
        t = elapsed[np.arange(len(out)), packets - 1]
        for answers, m in [("blocks", (out == "High") & (y == 1)), ("any answer", np.ones(len(out), bool))]:
            rows.append({"model": name, "K": K, "answers": answers, "segments": int(m.sum()),
                         "packets, mean": packets[m].mean() if m.any() else np.nan,
                         "seconds, median": np.median(t[m]) if m.any() else np.nan,
                         "seconds, 90th percentile": np.quantile(t[m], 0.9) if m.any() else np.nan})
decision_time = pd.DataFrame(rows).set_index(["model", "K", "answers"])
decision_time.to_csv(OUT / f"{TRACK}_decision_time.csv")
show(decision_time.round(4))


# %% [markdown]
# ## Forced answers and ceilings on the same rows
#
# A ceiling bounds a model only when identical rows are identical *in what the model reads*. The "headers" key is exactly
# the input of the model without TTL, destination port and IAT, so its ceilings are upper bounds for that model. No key
# holds TTL and destination port, so for the main model (and the baselines, which read its inputs) the ceilings are
# reference values only: a model that reads more fields can separate rows the key treats as identical and score above them.


# %%
PREFIXES = [1, 2, 3, 5, 10, 20, 50]
rows = []
for name, u in units.items():
    for p in ("val", "test"):
        idx = parts[p]
        for view, m in [("all", np.ones(len(idx), bool)), ("unseen", unseen_of[p])]:
            for k in PREFIXES:
                pred = (at_prefix(u["prob"][p], u["n"][p], k)[m] >= 0.5).astype(int)
                r = {"model": name, "split": p, "view": view, "k": k, **scores(u["y"][p][m], pred)}
                for key, H in [("headers + IAT", prefix_timed), ("headers", prefix_header)]:
                    c = ceiling(H[idx[m], k - 1], u["y"][p][m])
                    r.update({f"{cname}, {key}": value for cname, value in c.items() if cname != "rows"})
                r["rows"] = int(m.sum())
                rows.append(r)
forced = pd.DataFrame(rows).set_index(["model", "split", "view", "k"])
forced.to_csv(OUT / f"{TRACK}_forced_and_ceilings.csv")
no_testbed = selection["models"]["without TTL, destination port and IAT"][0]
# The model without the testbed group, against its exact ceilings (key "headers")
show(forced.loc[(no_testbed, "test"), ["rows", "accuracy", "MaxAcc, headers", "macro F1", "MaxF1, headers", "FPR",
                                  "minimum FPR, headers", "missed attacks", "minimum missed attacks, headers"]].round(4))


# %%
# The main model, with the ceilings as reference values (the keys leave out TTL and destination port)
show(forced.loc[(main["model"], "test"), ["rows", "accuracy", "MaxAcc, headers + IAT", "macro F1", "MaxF1, headers + IAT", "FPR",
                                     "minimum FPR, headers + IAT", "missed attacks", "minimum missed attacks, headers + IAT"]].round(4))


# %% [markdown]
# ## Calibration on test
#
# Temperature scaling divides the logit by one temperature per range of packets seen, fitted on the first validation half
# and capped at T >= 1 so that calibration never makes the model more confident. Each model is scored with the
# temperatures it uses (capped) and, for comparison, with the fitted ones (uncapped), at decision packets 1, 3, 10 and 50.
# ECE: 15 equal-width bins of P(attack), the gap between mean P(attack) and the share of attacks in each bin, weighted by
# the bin's size. The reliability bins at K = 50 are saved for the reliability diagram.


# %%
def calibration(p, y, bins=15):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    which = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean() for b in range(bins) if (which == b).any())
    return {"ECE": ece, "Brier": np.mean((p - y) ** 2), "NLL": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))}


fitted_T = pd.DataFrame({name: dict(zip(BUCKETS, u["fitted temperatures"][[list(STEP_BUCKET).index(b) for b in range(len(BUCKETS))]]))
                         for name, u in units.items()}).T
print("fitted temperatures (before the cap), per range of packets seen:")
print(fitted_T.round(3))
rows, reliability = [], []
for name, u in units.items():
    for reading, T in [("capped (used)", u["temperatures"]), ("fitted, uncapped", u["fitted temperatures"])]:
        prob = p_attack(u["z"]["test"], T)
        for view, m in [("all", np.ones(len(u["y"]["test"]), bool)), ("unseen", unseen_of["test"])]:
            for k in [1, 3, 10, L]:
                p, y = at_prefix(prob, u["n"]["test"], k)[m], u["y"]["test"][m]
                rows.append({"model": name, "temperature": reading, "view": view, "k": k, **calibration(p, y)})
                if k == L and view == "all":
                    which = np.minimum((p * 15).astype(int), 14)
                    reliability += [{"model": name, "temperature": reading, "bin": b, "segments": int((which == b).sum()),
                                     "mean P(attack)": p[which == b].mean(), "attack share": y[which == b].mean()}
                                    for b in range(15) if (which == b).any()]
calib = pd.DataFrame(rows).set_index(["model", "temperature", "view", "k"])
calib.to_csv(OUT / f"{TRACK}_calibration.csv")
pd.DataFrame(reliability).to_csv(OUT / f"{TRACK}_reliability.csv", index=False)
show(calib.xs("all", level="view").round(4))


# %% [markdown]
# Does the cap change the answers? With the uncapped temperatures, the High/Low cut-offs are retuned on the second
# validation half (same precision targets, same OOD flags) and the test answers are compared with the ones in use.


# %%
rows = []
for name, u in units.items():
    prob = {p: p_attack(u["z"][p], u["fitted temperatures"]) for p in ("thr", "test")}
    for K in [1, 3, 10, L]:
        flags = u["tuned"][K]["flags"]
        t = tune_targets(prob["thr"], flags["thr"], u["n"]["thr"], u["y"]["thr"], K)
        uncapped = route(prob["test"], flags["test"], u["n"]["test"], t["tau_attack"], t["tau_benign"], K)
        for reading, (out, packets) in [("capped (used)", decide(u, "test", K)), ("fitted, uncapped", uncapped)]:
            for view, m in [("all", np.ones(len(out), bool)), ("unseen", unseen_of["test"])]:
                rows.append({"model": name, "temperature": reading, "K": K, "view": view,
                             **known_metrics(out[m], u["y"]["test"][m], packets[m])})
cap_answers = pd.DataFrame(rows).set_index(["model", "temperature", "K", "view"])
cap_answers.to_csv(OUT / f"{TRACK}_cap_answers.csv")
show(cap_answers.xs("all", level="view")[["coverage", "attacks blocked", "block precision", "attacks called benign", "Low precision",
                                     "Medium", "mean packets to block"]].round(4))


# %% [markdown]
# ## Held-out captures: Mirai (another network) and ARP MitM (same network)
#
# On held-out traffic the only correct answer is "I don't know" (Medium or OOD); any High or Low answer is a failure, even
# when it matches the label. Mirai's labels are time-based (everything after packet 121,622 counts as attack), so its groups
# are read as before and during infection.


# %%
group_of = held.loc[parts["held"], "group"].to_numpy()
cap_of = held.loc[parts["held"], "capture"].to_numpy()
label_of = main["y"]["held"]
rows = []
for name, u in units.items():
    out, _ = decide(u, "held", L)
    for view, vm in [("all", np.ones(len(out), bool)), ("unseen", unseen_of["held"])]:
        for capture in ["Mirai", "ARP MitM"]:
            for lab, desc in [(0, "before infection" if capture == "Mirai" else "benign"), (1, "during infection" if capture == "Mirai" else "attack")]:
                m = vm & (cap_of == capture) & (label_of == lab)
                o = out[m]
                rows.append({"model": name, "view": view, "traffic": f"{capture}, {desc}", "segments": int(m.sum()),
                             "failure rate": np.isin(o, ["High", "Low"]).mean(), **{a: (o == a).mean() for a in ["High", "Low", "Medium", "OOD"]}})
held_table = pd.DataFrame(rows).set_index(["model", "view", "traffic"])
held_table.to_csv(OUT / f"{TRACK}_held_out.csv")
show(held_table.loc[main["model"]].round(4))


# %%
# OOD separation: AUROC of each detector's running mean, known test packets against held-out packets, per prefix bucket
real_t, real_h = real_steps(main["n"]["test"]), real_steps(main["n"]["held"])
rows = []
for detector in kl.DETECTORS:
    s = {p: main["mean"][p][detector] for p in ("test", "held")}
    for group in pd.unique(group_of):
        for b, bucket in enumerate(BUCKETS):
            inb = (STEP_BUCKET == b)[None, :]
            known_s = s["test"][real_t & inb]
            new_s = s["held"][(group_of == group)[:, None] & real_h & inb]
            if len(new_s):
                rows.append({"detector": detector, "held-out": group, "bucket": bucket,
                             "AUROC": roc_auc_score(np.r_[np.zeros(len(known_s)), np.ones(len(new_s))], np.r_[known_s, new_s])})
ood_auc = pd.DataFrame(rows).set_index(["detector", "held-out", "bucket"])["AUROC"].unstack("bucket")[list(BUCKETS)]
ood_auc.to_csv(OUT / f"{TRACK}_ood_auroc.csv")
show(ood_auc.round(3))


# %% [markdown]
# Which detector flags what, at K = 50: the share of segments flagged by the forest only, Mahalanobis only, both, and
# either (the OOD answer). Known test traffic shows the cost, the held-out captures what each detector adds.


# %%
rows = []
for name, u in units.items():
    t = u["tuned"][L]
    for p, groups in [("test", np.where(u["y"]["test"] == 1, "known test, attack", "known test, benign")), ("held", group_of)]:
        live = real_steps(u["n"][p])
        by = {d: ((u["mean"][p][d] > np.asarray(t["ood limits"][d])[STEP_BUCKET][None, :]) & live).any(axis=1) for d in kl.DETECTORS}
        for g in pd.unique(groups):
            m = groups == g
            f, mh = by["forest"][m], by["mahalanobis"][m]
            rows.append({"model": name, "traffic": g, "segments": int(m.sum()), "forest only": (f & ~mh).mean(),
                         "Mahalanobis only": (mh & ~f).mean(), "both": (f & mh).mean(), "either": (f | mh).mean()})
detector_overlap = pd.DataFrame(rows).set_index(["model", "traffic"])
detector_overlap.to_csv(OUT / f"{TRACK}_ood_detectors.csv")
show(detector_overlap.round(4))


# %% [markdown]
# ## Baselines: k-NN and XGBoost against the GRU
#
# One section, one training per model: k-NN (L1 distance, 15 neighbours, share of attack among them) and weighted
# XGBoost, at each checkpoint k on the first k packets of the main model's inputs. Every model is read at the
# same checkpoints with the OOD gate off, gets its High/Low cut-offs from the same precision targets on the same rows of
# the second validation half, and is scored on the same 15,000 test rows. Forced answers (P(attack) >= 0.5) sit next to the
# ceilings on those rows; the H/M/L reading is the main comparison. These models read TTL and destination port, which the
# ceiling key leaves out, so the ceilings here are reference values, not upper bounds.


# %%
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

CHECKS, EVAL_ROWS, KNN_TRAIN, NEIGHBOURS = [1, 2, 3, 5, 10, 20, L], 15_000, 40_000, 15
rng = np.random.default_rng(SEED)
pick = {p: np.sort(rng.choice(len(parts[p]), min(EVAL_ROWS, len(parts[p])), replace=False)) for p in ("thr", "test")}
cols = main["columns"]
cache = OUT / f"{TRACK}_baseline_probs.npz"
if cache.exists():
    probs = dict(np.load(cache))
else:
    x_tr, y_tr = load(X, part["train"], stats, cols), y_all[part["train"]]
    knn_rows = rng.permutation(len(x_tr))[:KNN_TRAIN]
    x_ev = {p: load(X, parts[p][pick[p]], stats, cols) for p in pick}
    probs = {f"{m}|{p}": np.zeros((len(pick[p]), len(CHECKS))) for m in ["k-NN", "XGBoost"] for p in pick}
    for j, k in enumerate(CHECKS):
        t0 = time.time()
        flat = x_tr[:, :k].reshape(len(x_tr), -1)
        nn = NearestNeighbors(n_neighbors=NEIGHBOURS, metric="manhattan", algorithm="brute", n_jobs=-1).fit(flat[knn_rows])
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=SEED,
                            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum()).fit(flat, y_tr)
        for p, x in x_ev.items():
            f = x[:, :k].reshape(len(x), -1)
            probs[f"k-NN|{p}"][:, j] = y_tr[knn_rows][nn.kneighbors(f, return_distance=False)].mean(axis=1)
            probs[f"XGBoost|{p}"][:, j] = xgb.predict_proba(f)[:, 1]
        print(f"k={k}: {time.time() - t0:.0f}s")
    np.savez(cache, **probs)
for p in pick:  # the GRU read at the same checkpoints
    probs[f"GRU|{p}"] = np.stack([at_prefix(main["prob"][p][pick[p]], main["n"][p][pick[p]], k) for k in CHECKS], axis=1)

rows, forced_rows = [], []
y_ev = {p: main["y"][p][pick[p]] for p in pick}
full = {p: np.full(len(pick[p]), len(CHECKS)) for p in pick}
no_flags = {p: np.zeros((len(pick[p]), len(CHECKS)), bool) for p in pick}
test_idx = parts["test"][pick["test"]]
test_unseen = unseen_of["test"][pick["test"]]
for m in ["GRU", "k-NN", "XGBoost"]:
    t = tune_targets(probs[f"{m}|thr"], no_flags["thr"], full["thr"], y_ev["thr"], len(CHECKS))
    out, _ = route(probs[f"{m}|test"], no_flags["test"], full["test"], t["tau_attack"], t["tau_benign"], len(CHECKS))
    for view, vm in [("all", np.ones(len(out), bool)), ("unseen", test_unseen)]:
        rows.append({"model": m, "view": view, "tau_attack": t["tau_attack"], "tau_benign": t["tau_benign"],
                     **known_metrics(out[vm], y_ev["test"][vm])})
        for j, k in enumerate(CHECKS):
            r = {"model": m, "view": view, "k": k, **scores(y_ev["test"][vm], (probs[f"{m}|test"][vm, j] >= 0.5).astype(int))}
            c = ceiling(prefix_timed[test_idx[vm], k - 1], y_ev["test"][vm])
            forced_rows.append({**r, **{f"{name} (same rows)": value for name, value in c.items()}})
baselines = pd.DataFrame(rows).set_index(["model", "view"])
baselines_forced = pd.DataFrame(forced_rows).set_index(["model", "view", "k"])
baselines.to_csv(OUT / f"{TRACK}_baselines.csv")
baselines_forced.to_csv(OUT / f"{TRACK}_baselines_forced.csv")
show(baselines[["coverage", "attacks blocked", "block precision", "attacks called benign", "benign called Low", "Low precision",
           "accuracy when answering", "macro F1 when answering", "Medium"]].round(4))


# %%
show(baselines_forced[["accuracy", "MaxAcc (same rows)", "macro F1", "MaxF1 (same rows)", "FPR", "missed attacks"]].round(4))


# %% [markdown]
# ## Main model vs without TTL, destination port and IAT


# %%
roles = {n: role for role, names in selection["models"].items() for n in names}
compare = pd.concat({view: pipeline.xs(("test", L, view), level=["split", "K", "view"])[shown] for view in ["all", "unseen"]},
                    names=["view"]).rename(index=roles, level=1)
compare.to_csv(OUT / f"{TRACK}_testbed_comparison.csv")
show(compare.round(4))


# %% [markdown]
# ## Evasion: perturbing what an attacker controls
#
# Test attack segments with timing jitter (each inter-arrival time multiplied by a factor in [0.8, 1.2]) or padding (up to
# 32 bytes added to each frame), with no retraining. A block that turns into Medium or OOD fails safe; one that turns into
# Low is the dangerous failure.


# %%
rng = np.random.default_rng(SEED)
att = parts["test"][main["y"]["test"] == 1]
raw = np.array(X[att])
mask = raw[:, :, kl.MASK_IDX] > 0
variants = {"none": raw}
v = raw.copy(); iat = np.expm1(v[:, :, kl.COL["iat"]]) * rng.uniform(0.8, 1.2, mask.shape); v[:, :, kl.COL["iat"]] = np.where(mask, np.log1p(iat), 0); variants["IAT jitter +/-20%"] = v
v = raw.copy(); size = np.expm1(v[:, :, kl.COL["frame_len"]]) + rng.integers(0, 33, mask.shape); v[:, :, kl.COL["frame_len"]] = np.where(mask, np.log1p(size), 0); variants["padding up to 32 bytes"] = v
mu, sd = main["stats"]
n_att = lengths[att]
t = main["tuned"][L]
rows = []
for vname, raw_v in variants.items():
    x = raw_v.copy()
    x[:, :, kl.CONT_IDX] = (x[:, :, kl.CONT_IDX] - mu) / sd
    x *= x[:, :, [kl.MASK_IDX]]
    z, h = predict(main["net"], x[:, :, main["columns"]].copy(), with_hidden=True)
    flags = kl.ood_flags(kl.ood_running_means(main["detectors"], h, n_att), t["ood limits"])
    del h
    out, _ = route(p_attack(z, main["temperatures"]), flags, n_att, t["tau_attack"], t["tau_benign"], L)
    rows.append({"perturbation": vname, **{a: (out == a).mean() for a in ["High", "Low", "Medium", "OOD"]}})
evasion = pd.DataFrame(rows).set_index("perturbation")
evasion.to_csv(OUT / f"{TRACK}_evasion.csv")
print(f"{len(att):,} test attack segments; inputs used: {main['groups']}")
show(evasion.round(4))


# %% [markdown]
# ## Deployment cost


# %%
hidden, layers = GRU_GRID[arch]
net = main["net"].eval()
torch.set_num_threads(1)
one = torch.from_numpy(load(X, parts["test"][:1], stats, main["columns"])[:, :1])
state, t0 = None, time.perf_counter()
with torch.no_grad():
    for _ in range(5000):
        o_, state = net.gru(one, state)
        net.head(o_)
per_packet_us = (time.perf_counter() - t0) / 5000 * 1e6
torch.set_num_threads(os.cpu_count())
cost = {"inference per packet, one thread (us)": per_packet_us, "packets per second, one thread": 1e6 / per_packet_us,
        "state per flow (bytes)": layers * hidden * 4 + 8, "forest trees": 200}
kl.save_json(cost, OUT / f"{TRACK}_deployment_cost.json")
show(cost)
